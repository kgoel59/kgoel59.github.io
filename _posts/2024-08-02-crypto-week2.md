---

layout: post  
title: "Crypto Week 2"  
date: 2024-08-02 15:09:00  
description:  Towards Modern Crypto"  
tags: projects learning uow
categories: learning  
giscus_comments: true  
featured: false  

---


Modern Crypto

shannon perfect cipher

message xor key =>  cipheher
cipheher xor key => message

Perfect Secrecy: A cryptographic system is said to achieve perfect secrecy if the probability distribution of the plaintext, given the ciphertext, is the same as the a priori probability distribution of the plaintext. In simpler terms, even if an attacker has the ciphertext, they gain no additional information about the plaintext than they had before seeing the ciphertext.

Pr[M=m]=Pr[M=m∣C=c]
This equation means that the probability of any message 
m being the actual plaintext is the same whether or not the attacker has seen the ciphertext c. Thus, observing the ciphertext provides no additional advantage in guessing the plaintext.

Def 2:
For every pairs of messages m0, m1 in the message space M, 
and every ciphertext c in the ciphertext space C
That means, the probability that C = c is the same for M = m0
or M = m1

We can

Theorem (Shannon)
In a system with perfect secrecy the number of keys is at least equal to 
the number of messages

[Lecture 2](/assets/pdf/crypto/2.%20CSCI471971_Cryptographic%20Notions.pdf)