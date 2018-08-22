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
	"github.com/pkg/errors"
	quad "github.com/fentec-project/gofe/quadratic"
	"fmt"
)

// This is a demonstration of how the SGP FE scheme for evaluation of
// quadratic multivariate polynomials can be used
// to evaluate a machine learning function on encrypted data.
//
// First, we assume that we learned a function for recognizing numbers
// from images using TensorFlow, and saved the parameters to files
// mat_valid.txt, mat_diag.txt and mat_proj.txt, which we put in the
// testdata folder.
func main() {
	// Read input matrices.
	// Matrices were exported from tensor flow.
	// We determine parameters like the number of vectors, vector space,
	// number of classes to be predicted and the number of examples, from the
	// input matrices.

	// Projection matrix - our data.
	// Contains nVecs vectors projected onto vector space of size vecSize.
	proj, err := readMatFromFile("testdata/mat_proj.txt")
	if err != nil {
		panic(errors.Wrap(err, "error reading projection matrix"))
	}
	nVecs := proj.Rows()
	vecSize := proj.Cols()

	// Diagonal matrix
	// number of rows of this matrix represents the number of classes.
	// The function will predict one of these classes.
	diag, err := readMatFromFile("testdata/mat_diag.txt")
	if err != nil {
		panic(errors.Wrap(err, "error reading diagonal matrix"))
	}
	nClasses := diag.Rows()
	if diag.Cols() != nVecs {
		panic(fmt.Sprintf("diagonal matrix must have %d columns", nVecs))
	}

	// Valid matrix
	// number of rows of this matrix represents the number of examples.
	valid, err := readMatFromFile("testdata/mat_valid.txt")
	if err != nil {
		panic(errors.Wrap(err, "error reading valid matrix"))
	}
	if valid.Cols() != vecSize {
		panic(fmt.Sprintf("valid matrix must have %d columns", vecSize))
	}

	// We know that all the values in the matrices are in the
	// interval [-bound, bound].
	bound := big.NewInt(100)

	// q is an instance of the FE scheme for quadratic multi-variate
	// polynomials constructed by Sans, Gay, Pointcheval (SGP)
	q := quad.NewSGP(vecSize, bound)

	// we generate a master secret key that we will need for encryption
	// of our data.
	fmt.Println("Generating master secret key...")
	msk, err := q.GenerateMasterKey()
	if err != nil {
		panic(errors.Wrap(err, "error when generating master keys"))
	}

	// First, we encrypt the data from mat_valid.txt
	// with our master secret key.
	// x = first row of matrix valid
	// y = also the first row of matrix valid
	fmt.Println("Encrypting...")
	c, err := q.Encrypt(valid[0], valid[0], msk)
	if err != nil {
		panic(errors.Wrap(err, "error when encrypting"))
	}

	// Then, we manipulate the encryption to be the encryption of the
	// projected data.
	// Note that this can also be done without knowing the secret key.
	fmt.Println("Manipulating encryption...")
	projC := projectEncryption(c, proj)

	fmt.Println("Manipulating secret key...")
	projSecKey := projectSecKey(msk, proj)

	// We create a new (projected) scheme instance for decrypting
	newBound := big.NewInt(1500000000)
	fmt.Println("Creating new (projected) scheme instance for decrypting...")
	qProj := quad.NewSGP(nVecs, newBound)

	res := make([]*big.Int, nClasses)
	maxValue := new(big.Int).Set(newBound)
	maxValue = maxValue.Neg(maxValue)

	fmt.Println("Predicting...")
	predictedNum := 0 // the predicted number
	for i := 0; i < nClasses; i++ {
		// We construct a diagonal matrix D that has the elements in the
		// current row of matrix diag on the diagonal.
		D := diagMat(diag[i])

		// We derive a feKey for obtaining the prediction from the encryption.
		// We will use this feKey for decrypting the final result,
		// e.g. x^T * D * y.
 		feKey, err := qProj.DeriveKey(projSecKey, D)
		if err != nil {
			panic(errors.Wrap(err, "error when deriving FE key"))
		}

		// We decrypt the encryption with the derived key feKey.
		// The result of decryption holds the value of x^T * D * y,
		// which in our case predicts the number from the handwritten
		// image.
		dec, err := qProj.Decrypt(projC, feKey, D)
		if err != nil {
			panic(errors.Wrap(err, "error when decrypting"))
		}
		res[i] = dec
		if dec.Cmp(maxValue) > 0 {
			maxValue.Set(dec)
			predictedNum = i
		}
	}

	fmt.Println("Prediction vector:", res)
	fmt.Println("The model predicts that the number on the image is", predictedNum)
}
