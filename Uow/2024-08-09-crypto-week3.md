---

layout: post  
title: "CSCI971 - Modern Cryptography Week 3"  
date: 2024-08-09 15:09:00  
description:  Block Cyphers and Symetric Key Encription"  
tags: projects learning uow
categories: learning  
giscus_comments: true  
featured: false  

---


How to construct symmetric key encription

We will use computer complexity and check if scheme is secure

Roadmap
Classic Cipher = Ceaser Cipher (algo based)
One time pad = Shannon

How to design a block cipher

Confusion and Diffusion: Design Principles

Confusion = if a single bit key is changed, cypher text is significantly different
Diffusion = if a single bit in plaintext is changed, cypher text is significantly different

Change in plain text and and key result significant change in cypher text


This is called avalanche effect

Block Cipher
Message -> Enc(key) -> Cipher -> Dec(key) -> Message

Typical block size 64 bits 128 bits

Common Design Approaches
iterated cipher => each iteration is called round. The output of each round is input to next round


DES = 16 rounds AES = 10 rounds

Each Round inside AES is: Subsitution Permutation network

DES (Data Encryption Standard)

why it is developed
1970 need standard encryption scheme

Standard need to have following properties

- high level security
- accordance with Kirchoff's law
- economic
- adaptable



DES (64 bit input, 64 bit output, 56 bit key)

16 round Feistel Network

Feistel Network

introduced by Feistel


how it work

original input - divide into two part
L0- 32 bit
R0 - 32 bit

(check week 3 slides)
R1 = R0 -> F(K) -> XOR L0
L1 = R0


Last Round

LN = R(N-1)
RN = R(N-1) -> F(K) -> XOR L(N-1)

S-box

An S-box is a basic component in symmetric key algorithms, used to perform substitution. It takes an input of a fixed size (a string of bits) and transforms it into an output of a fixed size. This substitution process is nonlinear, which is essential for the cryptographic strength of the algorithm

Des shortcomming 
2^56 keys = 10^17 keys (very short range keys)

we can test keys in parallel 

3DES


AES (Good enough for today)
(I think there was a video in youtube)
Does not use Feistel

Is a blockcipher a secure symmetric-key encryption
NO

it preserve statistacilly property of plain text (if you use it directly)

cypher text space should be much larger than plain text

Solution
Randomize
Choose ramdom IV
M' = IV XoR M
C <- Enc(K,M')


We say that the encryption is secure if no P.P.T adversary can win with a probability of 
½+1/poly(λ)

The underlying blockcipher is assumed to be a 
pseudorandom function (PRF), i.e., outputs of the 
blockcipher is indistinguishable from random values.


Encrypt long message



Electronic Code Book
Divide in block run cypher


CBC (Cypher block chain)
Initialize vector

CFB (Cypher Feedback )
Cypher of one block use as input to other not IV

Output feedback mode (OFB)



[Lecture 3](/assets/pdf/crypto/3.%20CSCI471971%20Symmetric-key%20Encryption.pdf)