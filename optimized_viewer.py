import argparse
import logging
import numpy as np
import pyqtgraph as pg
import argparse
import logging
import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.dockarea import DockArea, Dock
from scipy.signal import welch
import time 
import mne
#from mne_bids import BIDSPath
import os
import json
import serial 

# timing fx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.selected_trigger_channel = 14
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.erp_update_ms = 1000
        self.window_size = 10
        self.num_points = self.window_size * self.sampling_rate
        self.max_buffer_size = 2500
        self.sample_timestamps = np.full(self.max_buffer_size, np.nan)
        self.eeg_ring_buffer = np.full((len(self.exg_channels), self.max_buffer_size), np.nan)
        self.global_sample_count = 0
        self.plot_delay_sec = 0.5  # delay plotting by 0.5 seconds
        self.plotting_delay = 0.0 # delay of viewer

        # triggering stuff from the digital input pin
        self.last_trigger_value = 0
        self.last_rising_edge_time = 0
        self.min_trigger_interval = 1.0  # seconds to prevent duplicate ERP extraction


        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('BrainFlow Real-Time EEG Viewer with Docks')
        self.win.resize(1200, 800)
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        self.sett_doc = Dock("Settings", size=(800, 50))
        self.ts_dock = Dock("Time Series", size=(800, 600))
        self.psd_dock = Dock("Power Spectral Density", size=(400, 300))
        self.avg_ep_dock = Dock("Evoked Response", size=(400, 300))

        self.area.addDock(self.sett_doc, 'top')
        self.area.addDock(self.ts_dock, 'bottom', self.sett_doc)
        self.area.addDock(self.psd_dock, 'bottom', self.ts_dock)
        self.area.addDock(self.avg_ep_dock, 'right', self.psd_dock)

        self._init_settings()
        self._init_timeseries()
        self._init_psd()
        self._init_erp()

        self.ts_timer = QtCore.QTimer()
        self.ts_timer.timeout.connect(self.update_time_series)

        self.erp_timer = QtCore.QTimer()
        self.erp_timer.timeout.connect(self.update_psd_and_erp)

        self.trigger_lines = []
        self.trigger_times = []
        self.epochs = []

        self.max_epochs = 50
        self.win.closeEvent = self._handle_close

        # timing 
        self.update_timings = {
            "get_data": [],
            "buffer": [],
            "plot": [],
            "psd": [],
            "gui": [],
            "total": []}
        
        self.epoch_timings = []

    def _generate_colors(self, n):
        cmap = pg.colormap.get('CET-C1s')
        return [cmap.mapToQColor(i / max(1, n - 1)) for i in range(n)]

    def _init_settings(self):
        sett_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start Streaming")
        self.start_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.start_btn)

        self.end_btn = QtWidgets.QPushButton("Stop Streaming")
        self.end_btn.clicked.connect(self.stop_recording)
        layout.addWidget(self.end_btn)

        self.window_dropdown = QtWidgets.QComboBox()
        self.window_dropdown.addItems(["5", "10"])
        self.window_dropdown.setCurrentText(str(self.window_size))
        self.window_dropdown.currentTextChanged.connect(self.change_window_size)
        layout.addWidget(QtWidgets.QLabel("Window (s):"))
        layout.addWidget(self.window_dropdown)

        self.clear_erp_btn = QtWidgets.QPushButton("Clear ERP")
        self.clear_erp_btn.clicked.connect(self.clear_evoked_response)
        layout.addWidget(self.clear_erp_btn)

        self.test_trigger_btn = QtWidgets.QPushButton("Inject Test Pulse")
        self.test_trigger_btn.clicked.connect(self.trigger_prot)
        layout.addWidget(self.test_trigger_btn)

        # Dropdown to select which channel to display in bottom plot
        self.trigger_channel_dropdown = QtWidgets.QComboBox()
        self.available_extra_channels = list(range(BoardShim.get_num_rows(self.board_id)))  # All possible channel indices
        for ch in self.available_extra_channels:
            self.trigger_channel_dropdown.addItem(f"Channel {ch}", userData=ch)
        self.trigger_channel_dropdown.setCurrentIndex(14)  # Default to channel 13
        self.trigger_channel_dropdown.currentIndexChanged.connect(self.change_trigger_channel)
        layout.addWidget(QtWidgets.QLabel("Trigger Channel:"))
        layout.addWidget(self.trigger_channel_dropdown)

        sett_widget.setLayout(layout)
        self.sett_doc.addWidget(sett_widget)

    def start_recording(self):
        if not self.board_shim.is_prepared():
            logging.warning("Board not prepared! Cannot start recording.")
            return
        self.board_shim.start_stream(self.max_buffer_size, "")
        self.ts_timer.start(self.update_speed_ms)
        self.erp_timer.start(self.erp_update_ms)
        self.recording_start_time = QtCore.QTime.currentTime()
        self.start_btn.setEnabled(False)

    def stop_recording(self):
        if self.board_shim.is_prepared():
            self.board_shim.stop_stream()
            self.ts_timer.stop()
            self.erp_timer.stop()
            self.start_btn.setEnabled(True)
            for line in self.trigger_lines:
                for plot in self.plots:
                    plot.removeItem(line)
            self.save_erp()
            self.epochs.clear()
            self.trigger_times.clear()
            self.plot_timing_stats()
            
    def save_erp(self):
        if not self.epochs:
            logging.warning("No EPs found. EPs file is not being saved.")
        else:
            # Stack and shape epochs
            epochs_array = np.stack(self.epochs)
            n_epochs, n_channels, n_times = epochs_array.shape

            # Info and dummy events
            sfreq = self.sampling_rate
            ch_names = [f'EEG {i+1}' for i in range(n_channels)]
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

            events = np.column_stack((
                np.arange(n_epochs) * int(sfreq),
                np.zeros(n_epochs, dtype=int),
                np.ones(n_epochs, dtype=int)
            ))

            epochs = mne.EpochsArray(epochs_array, info, events=events, event_id={'stim': 1})

            # Paths
            bids_root = "./bids_dataset"
            deriv_root = os.path.join(bids_root, "derivatives", "erp_pipeline", "sub-01", "ses-01", "eeg")
            os.makedirs(deriv_root, exist_ok=True)

            # Filename according to BIDS derivatives convention
            deriv_file = "sub-01_ses-01_task-task_run-01_desc-epo_eeg.fif"
            save_path = os.path.join(deriv_root, deriv_file)

            # Save it
            epochs.save(save_path, overwrite=True)
            print(f"✅ Saved epoched EEG to: {save_path}")

            # Optionally write dataset_description.json for BIDS derivative
            desc_path = os.path.join(bids_root, "derivatives", "erp_pipeline", "dataset_description.json")
            if not os.path.exists(desc_path):
                desc_json = {
                    "Name": "ERP Pipeline",
                    "BIDSVersion": "1.8.0",
                    "PipelineDescription": {
                        "Name": "EEG Epoch Extraction",
                        "Version": "1.0"
                    }
                }
                with open(desc_path, "w") as f:
                    json.dump(desc_json, f, indent=4)
                print(f"📝 Created: {desc_path}")








            # # Stack epochs → shape: (n_epochs, n_channels, n_times)
            # epochs_array = np.stack(self.epochs)
            # n_epochs, n_channels, n_times = epochs_array.shape

            # # Create MNE Info
            # sfreq = self.sampling_rate
            # ch_names = [f'EEG {i+1}' for i in range(n_channels)]
            # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

            # # Dummy events: each epoch spaced 1s apart
            # events = np.column_stack((
            #     np.arange(n_epochs) * int(sfreq),
            #     np.zeros(n_epochs, dtype=int),
            #     np.ones(n_epochs, dtype=int)  # all epochs have event_id 1
            # ))

            # # Create Epochs object
            # epochs = mne.EpochsArray(epochs_array, info, events=events, event_id={'stim': 1})

            # # Define BIDS path to your raw data (assumes you saved raw earlier) but probably wont 

            # bids_root = "./bids_dataset"
            # deriv_root = os.path.join(bids_root, "derivatives", "erp_pipeline")
            

            # # Define derivative path
            # deriv_path = BIDSPath(
            #     subject='01',
            #     session='01',
            #     task='task',
            #     run='01',
            #     datatype='eeg',
            #     root=deriv_root,
            #     suffix='epo',
            #     extension='.fif',
            #     check = False
            # )

            # os.makedirs(deriv_path, exist_ok=True)


            # print('We here')
            # # Save epochs as derivative
            # epochs.save(deriv_path.fpath, overwrite=True)
            # print("✅ Epoched EEG saved to BIDS derivatives.")

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        self.colors = self._generate_colors(len(self.exg_channels))
        ts_widget = pg.GraphicsLayoutWidget()
        self.ts_dock.addWidget(ts_widget)

        for i in range(len(self.exg_channels)):
            p = ts_widget.addPlot(row=i, col=0)
            # Always show Y-axis (EEG amplitude)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            # Only show X-axis (time) on the last/bottom plot
            if i == len(self.exg_channels) - 1:
                p.showAxis('bottom', True)
                p.setMenuEnabled('bottom', False)
                p.setLabel('bottom', 'Time (s)')
            else:
                p.showAxis('bottom', False)

            if i == 0:
                p.setTitle('Time Series')
            p.setLabel('left', f'{i + 1}')
            p.enableAutoRange('y', True)
            self.plots.append(p)
            color = self.colors[i % len(self.colors)]
            curve = p.plot(pen=color)
            self.curves.append(curve)

        p = ts_widget.addPlot(row=len(self.exg_channels), col=0)
        p.showAxis('left', True)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', True)
        p.setMenuEnabled('bottom', False)
        p.setLabel('bottom', 'Time (s)')
        p.setLabel('left', 'Trig')
        p.enableAutoRange('y', True)
        self.plots.append(p)
        trigger_curve = p.plot(pen=pg.mkPen('r'))
        self.curves.append(trigger_curve)

    def _init_psd(self):
        psd_widget = pg.GraphicsLayoutWidget()
        self.psd_dock.addWidget(psd_widget)
        self.psd_plot = psd_widget.addPlot(title="Power Spectral Density (All Channels)")
        self.psd_plot.setLabel('left', 'Power (dB)')
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)')
        self.psd_curves = [self.psd_plot.plot(pen=self.colors[i]) for i in range(len(self.exg_channels))]

    def _init_erp(self):
        self.erp_widget = pg.GraphicsLayoutWidget()
        self.avg_ep_dock.addWidget(self.erp_widget)
        self.erp_plot = self.erp_widget.addPlot(title="Running Average Evoked Response")
        self.erp_plot.setLabel('bottom', 'Time (s)')
        self.erp_plot.setLabel('left', 'EEG (µV)')
        self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))

    def change_window_size(self, new_value):
        try:
            self.window_size = int(new_value)
            self.num_points = self.window_size * self.sampling_rate
            for plot in self.plots:
                plot.setXRange(-self.window_size, 0)
        except ValueError:
            logging.error(f"Invalid window size selected: {new_value}")

    def change_trigger_channel(self, index):
        self.selected_trigger_channel = self.trigger_channel_dropdown.itemData(index)

    def trigger_prot(self):
        if not hasattr(self, 'recording_start_time'):
            return
        elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        # Add a single trigger line to the bottom plot only
        
        line = pg.InfiniteLine(pos=elapsed, angle=90, pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashLine))
        self.plots[-1].addItem(line)
        self.trigger_lines.append(line)
        self.trigger_times.append(elapsed)
        QtCore.QTimer.singleShot(2000, lambda: self._extract_epoch(elapsed))


    def _handle_trigger(self, trigger_time):
        line = pg.InfiniteLine(pos=trigger_time, angle=90, pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashLine))
        self.plots[-2].addItem(line)
        self.trigger_lines.append(line)
        self.trigger_times.append(trigger_time)
        QtCore.QTimer.singleShot(2000, lambda: self._extract_epoch(trigger_time))


    def _extract_epoch(self, trigger_time):
        t0 = time.perf_counter()

        pre = int(0.5 * self.sampling_rate)
        post = int(1.0 * self.sampling_rate)
        eeg = self.eeg_ring_buffer.copy()
        timestamps = self.sample_timestamps.copy()
        valid_mask = ~np.isnan(timestamps)
        valid_times = timestamps[valid_mask]
        eeg = eeg[:, valid_mask]


        if eeg.shape[1] != valid_times.shape[0]:
            return

        closest_idx = np.argmin(np.abs(valid_times - trigger_time))
        start_idx = closest_idx - pre
        end_idx = closest_idx + post

        if start_idx < 0 or end_idx > valid_times.shape[0]:
            return
        
        epoch = eeg[:, start_idx:end_idx]
        baseline = np.mean(epoch[:, :pre], axis=1, keepdims=True)
        epoch -= baseline

        self.epochs.append(epoch)
        self._update_evoked_response(np.mean(np.stack(self.epochs), axis=0), pre + post)
        end = time.perf_counter()
        self.epoch_timings.append(end - t0)

    def _update_evoked_response(self, evoked_data, epoch_length):
        self.erp_plot.clear()
        self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))
        evoked_time = np.linspace(-0.5, 1.0, epoch_length)
        self.erp_plot.setTitle(f"Running Average Evoked Response (N={len(self.epochs)})")
        for ch_idx, ch_data in enumerate(evoked_data):
            self.erp_plot.plot(evoked_time, ch_data, pen=self.colors[ch_idx])

    def clear_evoked_response(self):
        self.epochs.clear()
        self.erp_plot.clear()
        self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))

    def _preprocess_signal(self, signal):
        DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(signal, self.sampling_rate, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        return signal

    def update_time_series(self):
            t0 = time.perf_counter()
            data = self.board_shim.get_current_board_data(self.max_buffer_size)
            t1 = time.perf_counter()

            available_samples = data.shape[1]
            if available_samples < 2:
                return

            num_points_to_use = min(self.num_points, available_samples)
            elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
            plot_time = elapsed - self.plot_delay_sec
            if plot_time <= 0:
                return  # not enough delay yet

            start_time = plot_time - (available_samples / self.sampling_rate)
            new_timestamps = np.linspace(start_time, plot_time, available_samples)

            for i in range(available_samples):
                idx = (self.global_sample_count + i) % self.max_buffer_size
                self.sample_timestamps[idx] = new_timestamps[i]
                self.eeg_ring_buffer[:, idx] = data[self.exg_channels, i]
            t2 = time.perf_counter()

            self.global_sample_count += available_samples


            start_time_for_window = max(0, plot_time - self.window_size)
            time_vector = np.linspace(start_time_for_window, plot_time, num_points_to_use)


            for count, channel in enumerate(self.exg_channels):
                signal = self._preprocess_signal(data[channel, -num_points_to_use:])
                self.curves[count].setData(time_vector, signal.tolist())
            t3 = time.perf_counter()



            trigger_signal = data[self.selected_trigger_channel, -num_points_to_use:]
            trigger_time_vector = time_vector  # same length as trigger_signal
            trigger_time_vector = time_vector  # same length as trigger_signal

            # Optional binary mapping if digital-type channel
            #if self.selected_trigger_channel == 13:
            #    trigger_signal = (trigger_signal > 0.5).astype(int)
            self.curves[-1].setData(time_vector, trigger_signal.tolist())

            # After self.curves[-1].setData(...)
            #trigger_signal = data[self.selected_trigger_channel, -num_points_to_use:]
            #trigger_time_vector = time_vector  # same length as trigger_signal

            # Check for rising edge (0 -> 1)
            for i in range(1, len(trigger_signal)):
                if trigger_signal[i - 1] == 0 and trigger_signal[i] == 1:
                    trigger_time = trigger_time_vector[i]
                    if trigger_time - self.last_rising_edge_time > self.min_trigger_interval:
                        self.last_rising_edge_time = trigger_time
                        self._handle_trigger(trigger_time)
                        break  # prevent multiple triggers per update



            for p in self.plots:
                p.setXRange(start_time_for_window, plot_time + self.plotting_delay, padding=0)

            self.app.processEvents()
            t4 = time.perf_counter()

            self.update_timings["get_data"].append(t1 - t0)
            self.update_timings["buffer"].append(t2 - t1)
            self.update_timings["plot"].append(t3 - t2)
            self.update_timings["psd"].append(0)
            self.update_timings["gui"].append(t4 - t3)
            self.update_timings["total"].append(t4 - t0)

    def _handle_close(self, event):
        if self.board_shim.is_prepared():
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            except Exception as e:
                logging.error(f"Error closing session: {e}")
        event.accept()

    def plot_timing_stats(self):
        iterations = range(len(self.update_timings["total"]))
        plt.figure(figsize=(12, 6))
        for key in ["get_data", "buffer", "plot", "psd", "gui"]:
            plt.plot(iterations, self.update_timings[key], label=key)
        plt.plot(iterations, self.update_timings["total"], label="total", linestyle='--', color='black')
        plt.axhline(y=0.05, color='red', linestyle=':', label='50ms Deadline')
        plt.title("Update Loop Timings")
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("update_timings.png")
        print("✅ Saved timing plot to update_timings.png")

        if self.epoch_timings:
            plt.figure(figsize=(10, 4))
            plt.plot(self.epoch_timings, label='Epoch Extraction Time')
            plt.axhline(y=0.05, color='red', linestyle=':', label='50ms Threshold')
            plt.title("Evoked Response Extraction Timings")
            plt.xlabel("Trigger Count")
            plt.ylabel("Time (s)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("epoch_extraction_timings.png")
            print("✅ Saved epoch extraction plot to epoch_extraction_timings.png")

    def update_psd_and_erp(self):
            data = self.board_shim.get_current_board_data(self.max_buffer_size)
            available_samples = data.shape[1]
            if available_samples < 2:
                return

            num_points_to_use = min(self.num_points, available_samples)
            for count, channel in enumerate(self.exg_channels):
                signal = data[channel, -num_points_to_use:]
                freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=self.sampling_rate)
                psd_db = 10 * np.log10(psd + 1e-12)
                self.psd_curves[count].setData(freqs, psd_db)

            if self.epochs:
                pre = int(0.5 * self.sampling_rate)
                self._update_evoked_response(np.mean(np.stack(self.epochs), axis=0), pre + int(1.0 * self.sampling_rate))



def main():


    def set_digital_input_mode(serial_port):
        try:
            ser = serial.Serial(serial_port, 115200, timeout=2)
            time.sleep(2)  # Let the board boot

            # Query current mode
            ser.write(b'//\n')  # Send double-slash to ask for board mode
            time.sleep(0.5)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"📋 Current mode: {response.strip()}")

            # Set to digital input mode
            ser.write(b'/3\n')
            time.sleep(0.5)

            # Query mode again
            ser.write(b'//\n')
            time.sleep(0.5)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"✅ Mode after /3: {response.strip()}")

            ser.close()
        except Exception as e:
            print(f"⚠️ Error: {e}")




    BoardShim.enable_dev_board_logger()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=0)
    parser.add_argument('--ip-port', type=int, default=0)
    parser.add_argument('--ip-protocol', type=int, default=0)
    parser.add_argument('--ip-address', type=str, default='')
    parser.add_argument('--serial-port', type=str, default='')
    parser.add_argument('--mac-address', type=str, default='')
    parser.add_argument('--other-info', type=str, default='')
    parser.add_argument('--streamer-params', type=str, default='')
    parser.add_argument('--serial-number', type=str, default='')
    parser.add_argument('--board-id', type=int, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--master-board', type=int, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    if args.serial_port:
        set_digital_input_mode(args.serial_port)
        
    else:
        print("⚠️ No serial port provided, skipping digital mode set.")
    

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()    
        graph = Graph(board_shim)
        graph.win.show()
        graph.app.exec()
    except Exception:
        logging.warning('Exception occurred', exc_info=True)
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()


if __name__ == '__main__':
    main()
