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


# timing fx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.erp_update_ms = 1000
        self.window_size = 10
        self.num_points = self.window_size * self.sampling_rate
        self.max_buffer_size = 3750
        self.sample_timestamps = np.full(self.max_buffer_size, np.nan)
        self.eeg_ring_buffer = np.full((len(self.exg_channels), self.max_buffer_size), np.nan)
        self.global_sample_count = 0

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
        self.window_dropdown.addItems(["5", "10", "15"])
        self.window_dropdown.setCurrentText(str(self.window_size))
        self.window_dropdown.currentTextChanged.connect(self.change_window_size)
        layout.addWidget(QtWidgets.QLabel("Window (s):"))
        layout.addWidget(self.window_dropdown)

        self.clear_erp_btn = QtWidgets.QPushButton("Clear ERP")
        self.clear_erp_btn.clicked.connect(self.clear_evoked_response)
        layout.addWidget(self.clear_erp_btn)

        self.test_trigger_btn = QtWidgets.QPushButton("Test Trigger")
        self.test_trigger_btn.clicked.connect(self.trigger_prot)
        layout.addWidget(self.test_trigger_btn)

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
            self.epochs.clear()
            self.trigger_times.clear()
            self.plot_timing_stats()

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

    def update(self):


    
        t0 = time.perf_counter()

        data = self.board_shim.get_current_board_data(self.max_buffer_size)
        t1 = time.perf_counter()

        available_samples = data.shape[1]
        if available_samples < 2:
            return

        num_points_to_use = min(self.num_points, available_samples)
        elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        start_time = elapsed - (available_samples / self.sampling_rate)
        new_timestamps = np.linspace(start_time, elapsed, available_samples)

        for i in range(available_samples):
            idx = (self.global_sample_count + i) % self.max_buffer_size
            self.sample_timestamps[idx] = new_timestamps[i]
            self.eeg_ring_buffer[:, idx] = data[self.exg_channels, i]
        t2 = time.perf_counter()

        self.global_sample_count += available_samples
        start_time_for_window = max(0, elapsed - self.window_size)
        time_vector = np.linspace(start_time_for_window, elapsed, num_points_to_use)

        for count, channel in enumerate(self.exg_channels):
            signal = self._preprocess_signal(data[channel, -num_points_to_use:])
            self.curves[count].setData(time_vector, signal.tolist())
        t3 = time.perf_counter()

        for count, channel in enumerate(self.exg_channels):
            signal = data[channel, -num_points_to_use:]
            freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=self.sampling_rate)
            psd_db = 10 * np.log10(psd + 1e-12)
            self.psd_curves[count].setData(freqs, psd_db)
        #t4 = time.perf_counter()

        for p in self.plots:
            p.setXRange(start_time_for_window, elapsed + 0.1, padding=0)

        t4 = time.perf_counter()
        self.app.processEvents()
        t5 = time.perf_counter()

        self.update_timings["get_data"].append(t1 - t0)
        self.update_timings["buffer"].append(t2 - t1)
        self.update_timings["plot"].append(t3 - t2)
        self.update_timings["psd"].append(t4 - t3)
        self.update_timings["gui"].append(t5 - t4)
        self.update_timings["total"].append(t5 - t0)

    def update_time_series(self):
            t0 = time.perf_counter()
            data = self.board_shim.get_current_board_data(self.max_buffer_size)
            t1 = time.perf_counter()

            available_samples = data.shape[1]
            if available_samples < 2:
                return

            num_points_to_use = min(self.num_points, available_samples)
            elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
            start_time = elapsed - (available_samples / self.sampling_rate)
            new_timestamps = np.linspace(start_time, elapsed, available_samples)

            for i in range(available_samples):
                idx = (self.global_sample_count + i) % self.max_buffer_size
                self.sample_timestamps[idx] = new_timestamps[i]
                self.eeg_ring_buffer[:, idx] = data[self.exg_channels, i]
            t2 = time.perf_counter()

            self.global_sample_count += available_samples
            start_time_for_window = max(0, elapsed - self.window_size)
            time_vector = np.linspace(start_time_for_window, elapsed, num_points_to_use)

            for count, channel in enumerate(self.exg_channels):
                signal = self._preprocess_signal(data[channel, -num_points_to_use:])
                self.curves[count].setData(time_vector, signal.tolist())
            t3 = time.perf_counter()

            for p in self.plots:
                p.setXRange(start_time_for_window, elapsed + 0.1, padding=0)

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
