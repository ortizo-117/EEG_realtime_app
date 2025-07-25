import argparse
import logging

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

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='Full EEG Buffer Plot', size=(800, 600), show=True)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(5000)  # Update every 1 second to avoid overload
        QtWidgets.QApplication.instance().exec()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        self.buffer_size = 15000  # Fixed max buffer size
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.setXRange(0, self.buffer_size, padding=0)
            p.enableAutoRange(axis='x', enable=False)
            if i != len(self.exg_channels) - 1:
                p.showAxis('bottom', False)
            else:
                p.setLabel('bottom', 'Samples')
            if i == 0:
                p.setTitle('EEG Buffer Plot')
            curve = p.plot(pen='y')
            self.plots.append(p)
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data (15000)
        print(f"Buffer length: {data.shape[1]} samples")
        #num_samples = data.shape[1]

        for i, channel in enumerate(self.exg_channels):
            channel_data = data[channel]
            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

            x_vals = list(range(len(channel_data)))
            self.curves[i].setData(x_vals, channel_data.tolist())

        self.app.processEvents()




def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
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
        board_shim.start_stream(15000, args.streamer_params)
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