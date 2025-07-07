import argparse
import logging
import numpy as np

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Time Series and Multi-Channel PSD', size=(800, 800), show=True)

        self._init_timeseries()
        self._init_psd()

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.win.closeEvent = self._handle_close
        self.app.exec()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []

        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']

        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Time Series')
            p.enableAutoRange('y', True)
            self.plots.append(p)

            color = self.colors[i % len(self.colors)]
            curve = p.plot(pen=color)
            self.curves.append(curve)

    def _init_psd(self):
        row = len(self.exg_channels)
        self.psd_plot = self.win.addPlot(row=row, col=0)
        self.psd_plot.setTitle('Power Spectral Density (All Channels)')
        self.psd_plot.setLabel('left', 'Power (dB)')
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)')
        self.psd_plot.setLogMode(x=False, y=False)
        self.psd_curves = []

        # Create one PSD curve per channel with matching color
        for i in range(len(self.exg_channels)):
            color = self.colors[i % len(self.colors)]
            curve = self.psd_plot.plot(pen=color, name=f'Ch {i}')
            self.psd_curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())

        # Compute and plot PSD for each channel
        for count, channel in enumerate(self.exg_channels):
            signal = data[channel]
            windowed = signal * np.hanning(len(signal))
            fft = np.fft.rfft(windowed)
            psd = np.abs(fft) ** 2
            psd_db = 10 * np.log10(psd + 1e-12)
            freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.sampling_rate)

            self.psd_curves[count].setData(freqs, psd_db)

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
        board_shim.start_stream(450000, args.streamer_params)
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()