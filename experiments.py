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
	- lstm tanh 1000
	- lstm relu 1000
	- gru tanh 1000
	- weight decay 0.002 tanh gru 1000
		Running:
		- 2000 gru
		- 2000 lstm

	P2 :
	- 1000 attn layers 2 gru
	- 2000 attn layers 2 gru
	- 2000 LSTM
	- 2000 GRU
		Running:
		- 