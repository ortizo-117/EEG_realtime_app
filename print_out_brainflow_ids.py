from brainflow.board_shim import BoardShim

board_id = 0  # or whatever you're using
BoardShim.enable_dev_board_logger()
\
print(BoardShim.get_board_descr(board_id))