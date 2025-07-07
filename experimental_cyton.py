import argparse
import logging
import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.dockarea import DockArea, Dock


class Graph:

    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 10
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('BrainFlow Real-Time EEG Viewer with Docks')
        self.win.resize(1200, 800)

        # Initialize DockArea as central widget
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        # Create docks
        self.sett_doc = Dock("Settings",size =(800, 50))
        self.ts_dock = Dock("Time Series", size=(800, 600))
        self.psd_dock = Dock("Power Spectral Density", size=(400, 300))
        self.avg_ep_dock = Dock("Evoked Response",size = (400,300))

        # Add docks to the dock area
        self.area.addDock(self.sett_doc,'top')
        self.area.addDock(self.ts_dock, 'bottom',self.sett_doc)
        self.area.addDock(self.psd_dock, 'bottom', self.ts_dock)
        self.area.addDock(self.avg_ep_dock, 'right', self.psd_dock)

        # Initialize plots in docks
        self._init_settings()
        self._init_timeseries()
        self._init_psd()
        self._init_erp()

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        # Epochs
        self.trigger_lines = []
        self.trigger_times = []
        self.epochs = []

        self.win.closeEvent = self._handle_close
        self.win.show()
        self.app.exec()


    def _init_settings(self):
        # Create a QWidget to hold controls
        sett_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()  # horizontal layout for buttons

        # Create Begin Recording button
        self.start_btn = QtWidgets.QPushButton("Start Streaming")
        self.start_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.start_btn)

        # Stop Recording Button
        self.end_btn = QtWidgets.QPushButton("Stop Streaming")
        self.end_btn.clicked.connect(self.stop_recording)
        layout.addWidget(self.end_btn)
        
        # Window Length dopdown
        self.window_dropdown = QtWidgets.QComboBox()
        self.window_dropdown.addItems(["5", "10", "15", "20"])  # window lengths in seconds
        self.window_dropdown.setCurrentText(str(self.window_size))  # set initial value
        self.window_dropdown.currentTextChanged.connect(self.change_window_size)
        layout.addWidget(QtWidgets.QLabel("Window (s):"))
        layout.addWidget(self.window_dropdown)


        self.clear_erp_btn = QtWidgets.QPushButton("Clear ERP")
        self.clear_erp_btn.clicked.connect(self.clear_evoked_response)
        layout.addWidget(self.clear_erp_btn)




        # test trigger button
        self.test_trigg_btn = QtWidgets.QPushButton("Test trigg")
        self.test_trigg_btn.clicked.connect(self.trigger_prot)
        layout.addWidget(self.test_trigg_btn)

        # Optional: add more controls later here...

        sett_widget.setLayout(layout)
        self.sett_doc.addWidget(sett_widget)

    def start_recording(self):
        try:
            # Check if the board is prepared before starting stream
            if not self.board_shim.is_prepared():
                logging.warning("Board not prepared! Cannot start recording.")
                return

            self.board_shim.start_stream(450000,"")
            self.timer.start(self.update_speed_ms)  # Start update timer
            self.recording_start_time = QtCore.QTime.currentTime()

            logging.info("Recording started.")
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")

        self.start_btn.setEnabled(False)

    def stop_recording(self):
        try:
            if self.board_shim.is_prepared():
                logging.info("Stopping recording...")

                # Stop the data stream if it's running
                self.board_shim.stop_stream()

                # Stop the GUI update timer
                self.timer.stop()

                # Optionally re-enable start button if you want to allow restarting
                self.start_btn.setEnabled(True)

                # Remove all trigger lines from plots
                for line in self.trigger_lines:
                    for plot in self.plots:
                        plot.removeItem(line)
                self.epochs.clear()
                self.trigger_times.clear()
                logging.info("All trigger lines cleared after stopping recording.")
                logging.info("Recording stopped.")
            else:
                logging.warning("Board is not prepared; nothing to stop.")
        except Exception as e:
            logging.error(f"Failed to stop recording: {e}")
        

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        cmap = pg.colormap.get('CET-C1s')
        self.colors = [
            cmap.mapToQColor(i / max(1, len(self.exg_channels) - 1))
            for i in range(len(self.exg_channels))]

        # Create GraphicsLayoutWidget and add to the Time Series dock
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

            p.setXRange(-self.window_size, 0)
            p.enableAutoRange('y', True)

            self.plots.append(p)

            color = self.colors[i % len(self.colors)]
            curve = p.plot(pen=color)
            self.curves.append(curve)


    def _init_psd(self):
        # Create GraphicsLayoutWidget and add to the PSD dock
        psd_widget = pg.GraphicsLayoutWidget()
        self.psd_dock.addWidget(psd_widget)

        self.psd_plot = psd_widget.addPlot()
        self.psd_plot.setTitle('Power Spectral Density (All Channels)')
        self.psd_plot.setLabel('left', 'Power (dB)')
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)')
        self.psd_plot.setLogMode(x=False, y=False)
        self.psd_curves = []

        for i in range(len(self.exg_channels)):
            color = self.colors[i % len(self.colors)]
            curve = self.psd_plot.plot(pen=color, name=f'Ch {i}')
            self.psd_curves.append(curve)


    def _init_erp(self):
        # Create the widget that will hold ERP plots
        self.erp_widget = pg.GraphicsLayoutWidget()
        self.avg_ep_dock.addWidget(self.erp_widget)

        # Create the initial plot inside the widget
        self.erp_plot = self.erp_widget.addPlot(title="Running Average Evoked Response")
        self.erp_plot.setLabel('bottom', 'Time (s)')
        self.erp_plot.setLabel('left', 'EEG (µV)')

        # Draw vertical line at time zero (trigger onset)
        self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))

        logging.info("Initialized ERP plot.")


        
    
    def change_window_size(self,new_value):
        try:
            new_window_size = int(new_value)
            self.window_size = new_window_size
            self.num_points = self.window_size * self.sampling_rate
            
            # update x range for all time- series plot
            for plot in self.plots:
                plot.setXRange(-self.window_size,0)
            logging.info(f"Updated sliding window size to {self.window_size} seconds.")

        except ValueError:
            logging.error(f"Invalid window size selected: {new_value}")

        
    def trigger_prot(self):
        if not hasattr(self, 'recording_start_time'):
            logging.warning("Cannot place trigger: recording has not started yet!")
            return

        elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        logging.info(f"Trigger placed at {elapsed:.2f} seconds since start.")

        # Add vertical lines on time-series plots immediately
        for plot in self.plots:
            line = pg.InfiniteLine(pos=elapsed, angle=90, pen=pg.mkPen('y', width=2, style=QtCore.Qt.DashLine))
            plot.addItem(line)
            self.trigger_lines.append(line)

        self.trigger_times.append(elapsed)

        #  Schedule ERP extraction ~1 second later
        QtCore.QTimer.singleShot(2000, lambda: self._extract_epoch(elapsed))

    def _extract_epoch(self, elapsed):
        pre_samples = int(0.5 * self.sampling_rate)
        post_samples = int(1.0 * self.sampling_rate)
        epoch_length = pre_samples + post_samples

        data_count = self.board_shim.get_board_data_count()
        full_data = self.board_shim.get_current_board_data(data_count)

        trig_sample_idx = int(elapsed * self.sampling_rate)
        start_idx = trig_sample_idx - pre_samples
        end_idx = trig_sample_idx + post_samples

        buffer_samples = full_data.shape[1]

        if start_idx < 0 or end_idx > buffer_samples:
            logging.warning(f"Trigger at {elapsed:.2f}s is out of bounds for ERP window: "
                            f"requested start_idx={start_idx}, end_idx={end_idx}, "
                            f"but buffer contains 0 to {buffer_samples - 1} samples.")
            return

        # Extract data: channels x epoch_length
        epoch = full_data[np.ix_(self.exg_channels, range(start_idx, end_idx))]

        # Baseline correction
        baseline = np.mean(epoch[:, :pre_samples], axis=1, keepdims=True)
        epoch -= baseline

        self.epochs.append(epoch)

        # Compute running average and update ERP plot
        self._update_evoked_response(np.mean(np.stack(self.epochs), axis=0), epoch_length)

        

    def _update_evoked_response(self, evoked_data, epoch_length):
        if not hasattr(self, 'erp_plot'):
            logging.warning("ERP plot not initialized.")
            return

        # Clear existing curves in the ERP plot
        self.erp_plot.clear()

        # Draw vertical line at time zero again (since clear() removes it)
        self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))

        evoked_time = np.linspace(-0.5, 1.0, epoch_length)

        # Update the plot's title to show the number of epochs
        num_epochs = len(self.epochs)
        self.erp_plot.setTitle(f"Running Average Evoked Response (N={num_epochs})")

        for ch_idx, ch_data in enumerate(evoked_data):
            self.erp_plot.plot(evoked_time, ch_data, pen=self.colors[ch_idx % len(self.colors)], name=f'Ch {ch_idx+1}')
            

    def clear_evoked_response(self):
        self.epochs.clear()
        # Clear existing curves from the persistent ERP plot
        if hasattr(self, 'erp_plot'):
            self.erp_plot.clear()
            # Redraw vertical line at time zero (since clear() removes it)
            self.erp_plot.addLine(x=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))

        logging.info("Cleared evoked response plot and reset epochs.")

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        available_samples = data.shape[1]

        if available_samples < 2:  # Sanity check: no data at all yet
            logging.warning("No samples received yet.")
            return

        if available_samples < self.num_points:
            logging.warning(f"Received fewer samples than expected: {available_samples}, "
                            f"displaying available samples instead.")
            num_points_to_use = available_samples
        else:
            num_points_to_use = self.num_points

        # Build time vector spanning -window_size to 0 with correct length
        #time_vector = np.linspace(-self.window_size, 0, num_points_to_use)
        elapsed = self.recording_start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        start_time_for_window = max(0, elapsed - self.window_size)
        time_vector = np.linspace(start_time_for_window, elapsed, num_points_to_use)

        # Update time-series plots
        for count, channel in enumerate(self.exg_channels):
            signal = data[channel, -num_points_to_use:]  # Get last N samples

            # Preprocessing: detrend and filter
            DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

            # Update time-series curve with time on X-axis
            self.curves[count].setData(time_vector, signal.tolist())
        
        # Update x-range to follow elapsed time
        for p in self.plots:
            p.setXRange(start_time_for_window, elapsed)


        # Update PSD plots
        for count, channel in enumerate(self.exg_channels):
            signal = data[channel, -num_points_to_use:]
            windowed = signal * np.hanning(len(signal))
            fft = np.fft.rfft(windowed)
            psd = np.abs(fft) ** 2
            psd_db = 10 * np.log10(psd + 1e-12)
            freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.sampling_rate)

            self.psd_curves[count].setData(freqs, psd_db)

        # Keep GUI responsive
        self.app.processEvents()


    def _handle_close(self, event):
        logging.info('Window closed. Releasing session.')
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        event.accept()



def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

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
        #board_shim.start_stream(450000, args.streamer_params)
        Graph(board_shim)
    except Exception:
        logging.warning('Exception occurred', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()