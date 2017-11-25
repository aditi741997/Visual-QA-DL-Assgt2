P1
	relu tanh (2 :( )
	lstm gru
	2000 (both lstm gru)
	weight decay / gamma : 
P2
	lstm gru
	2000 (both lstm gru)
	1000 gru attn
	2000 gru attn


	katta:
	P1 :
		- 2000 gru 			wt decay	 	log_2000_p1_gru.txt
		- 2000 lstm 		wt decay	 	log_2000_p1.txt
		- lstm tanh 1000	wt decay	 	log_1000_p1_lstm_tanh.txt
		- weight decay 0.0005 tanh gru 1000 log_1000_p1_gru_tanh_decay_nz.txt -> lower tr accu than 0 wt decay, same test accu : regularization
		- lstm relu 1000 					log_1000_p1_lstm_relu.txt
		- gru tanh 1000						log_1000_p1_gru_tanh.txt

	2000 -> with 0.01 : nothing.
			0.025 :
			0.04 :

	P2 :
	- 1000 attn layers 2 gru
	- 2000 attn layers 2 gru
	- 2000 LSTM
	- 2000 GRU
	0.01 : low
	0.05 : too high. missing minima.
	0.025 : 
	2000 -> parameters increase a lot, slow learning, so didnt do so well.
