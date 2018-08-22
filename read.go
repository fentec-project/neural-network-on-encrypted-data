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
	"bufio"
	"math/big"
	"os"
	"strings"

	"github.com/pkg/errors"
	"github.com/fentec-project/gofe/data"
)

// readMatFromFile reads matrix elements from the provided file
// and gives a matrix
func readMatFromFile(path string) (data.Matrix, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrap(err, "error reading matrix from file")
	}

	scanner := bufio.NewScanner(file)
	vecs := make([]data.Vector, 0)

	for scanner.Scan() {
		line := scanner.Text()
		numbers := strings.Split(line, " ")
		v := make(data.Vector, len(numbers))
		for i, n := range numbers {
			v[i], _ = new(big.Int).SetString(n, 10)
		}
		vecs = append(vecs, v)
	}

	return data.NewMatrix(vecs)
}
