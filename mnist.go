/*
Copyright (c) 2018 XLAB d.o.o

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"math/big"

	"github.com/fentec-project/gofe/data"
	quad "github.com/fentec-project/gofe/quadratic"
)

// innerProdEncryption accepts the encryption c of (x, y)
// and vectors u, v. It manipulates the encryption in order
// to obtain the encryption of (x*u, y*v).
func innerProdEncryption(c *quad.SGPCipher, u, v data.Vector) *quad.SGPCipher {
	zeros := make(data.Vector, 2)
	zeros[0] = big.NewInt(0)
	zeros[1] = big.NewInt(0)

	firstG1 := zeros.MulG1()
	secondG2 := zeros.MulG2()

	for i := 0; i < len(c.AMulG1); i++ {
		ui := make(data.Vector, 2)
		ui[0] = u[i]
		ui[1] = u[i]
		tmpG1 := ui.MulVecG1(c.AMulG1[i])
		firstG1 = firstG1.Add(tmpG1)

		vi := make(data.Vector, 2)
		vi[0] = v[i]
		vi[1] = v[i]
		tmpG2 := vi.MulVecG2(c.BMulG2[i])
		secondG2 = secondG2.Add(tmpG2)
	}

	first := make([]data.VectorG1, 1)
	first[0] = firstG1
	second := make([]data.VectorG2, 1)
	second[0] = secondG2

	return quad.NewSGPCipher(c.G1MulGamma, first, second)
}

// projectEncryption accepts the encryption c of (x, y)
// and projection matrix P. It manipulates the encryption
// in order to obtain an encryption of (P*x, P*y).
func projectEncryption(c *quad.SGPCipher, P data.Matrix) *quad.SGPCipher {
	n := len(P) // number of vectors in the P matrix
	aMulG1 := make([]data.VectorG1, n)
	bMulG2 := make([]data.VectorG2, n)

	for i := 0; i < n; i++ {
		innerC := innerProdEncryption(c, P[i], P[i])
		aMulG1[i] = innerC.AMulG1[0]
		bMulG2[i] = innerC.BMulG2[0]
	}

	return quad.NewSGPCipher(c.G1MulGamma, aMulG1, bMulG2)
}

// innerProdSecKey accepts the secret key msk that was used for
// encryption of (x, y), and vectors u, v.
// It manipulates the provided master secret key in order to obtain
// a secret key for encryption of (x*u, y*v).
// Secret key is manipulated with the inner (dot) product operation.
func innerProdSecKey(msk *quad.SGPSecKey, u, v data.Vector) *quad.SGPSecKey {
	s := make(data.Vector, 1)
	s[0], _ = msk.S.Dot(u)

	t := make(data.Vector, 1)
	t[0], _ = msk.T.Dot(v)

	return quad.NewSGPSecKey(s, t)
}

// projectSecKey accepts the secret key msk that was used for
// encryption of (x, y), and the projection matrix P.
// It manipulates the provided master secret key in order to obtain
// a secret key for encryption of (P*x, P*y).
func projectSecKey(msk *quad.SGPSecKey, P data.Matrix) *quad.SGPSecKey {
	n := len(P) // number of vectors in the P matrix
	s := make(data.Vector, n)
	t := make(data.Vector, n)

	for i := 0; i < n; i++ {
		secKeyInner := innerProdSecKey(msk, P[i], P[i])
		s[i] = secKeyInner.S[0]
		t[i] = secKeyInner.T[0]
	}

	return quad.NewSGPSecKey(s, t)
}

// diagMat takes a vector v and returns a diagonal matrix
// with elements of v on the diagonal.
func diagMat(v data.Vector) data.Matrix {
	l := len(v)
	mat := make(data.Matrix, l)

	for j, vi := range v {
		vec := make(data.Vector, l)
		for i := 0; i < l; i++ {
			vec[i] = big.NewInt(0)
			if i == j {
				vec[i].Set(vi)
			}
		}
		mat[j] = vec
	}

	return mat
}
