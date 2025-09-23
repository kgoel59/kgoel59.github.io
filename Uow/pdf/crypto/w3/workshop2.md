Question 1. Please show that the following encryption scheme is not IND-CCA secure, where Enc is a secure blockcipher:

1.Choose a random IV.
2.Run W←Enc(K, IV)
3.Compute C=W XOR M
4.Output CT=(IV, C)

1  

IV = test
Key = test
M = 0x68656C6C6F

W <- Enc(K,IV)
0x8ca64de9 <- Enc(test,test)

C <- W XOR M
0x68e9ca2186 <= 0x8ca64de9 ^ 0x68656C6C6F

CT=(IV, C)
0x30699b0e4485950e9e12b2b5 = Enc(test, 0x68e9ca2186)

2 challange text M0 M1

modify last bit

decrypted result have difference which can be used to get info on the underlying message