# EEG-ECoG Task

## What data is
See a Method section in this paper.
Oosugi, Naoya, et al. "A new method for quantifying the performance of EEG blind source separation algorithms by referencing a simultaneously recorded ECoG signal." Neural Networks 93 (2017): 1-6.

   
## Data Format
A. ECoG_{experiments}.mat
	Data matrix: Channel x Time
	Sampling rate: 1000Hz
	Location of electrodes:  Refer to 
		1. http://neurotycho.org/sites/default/files/images/brainmap-Chibi-v2.jpg 
		2. http://neurotycho.org/data/20110621ktmdspatialmapecogarraychibitoruyanagawa


B. EEG_{experiments}.mat
	Data matrix: Channel x Time
	Sampling rate: 1000Hz
	Location of electrodes: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, C4, T4, T5, P3, T6, O1, O2 (determined by 10-20 system)



{experiments} means state of the experiment. Refer to "2.2. Anesthesia experiment section".



[Author] Naoya Oosugi,Yasuo Nagasaka, Naomi Hasegawa 

