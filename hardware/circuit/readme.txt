MLA_opamp.pdf provides the circuit schematic.

mla_opamp.PcbDoc provides the PCB layout for fabrication.

lsi_pcb_ctrl/FPGA_part/EmptySketch/ contains the firmware for the Arduino MKR Vidor 4000. Compile and upload EmptySketch.ino to the board.

lsi_pcb_ctrl/run.bat is the entry point for communicating with the FPGA and sensor array. It runs the calibration and data acquisition routines in sequence.

The PCB uses a multiplexer for the DAC, with 'SIGNAL_SOURCE' selecting the input signal. The experiments reported in the paper use the output of the first amplifier, 'amp1_o_f'.