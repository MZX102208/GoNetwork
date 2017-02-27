package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"sync"
)

func main() {

	trainingData := initTrainingImages()
	trainingLabels := initTrainingLabels()
	testData := initTestImages()
	testLabels := initTestLabels()



	layerSizes := make([]int, 0)
	layerSizes = append(layerSizes, 188)
	//layerSizes = append(layerSizes, 20*26)
	layerSizes = append(layerSizes, 90)
	layerSizes = append(layerSizes, 44)
	//layerSizes = append(layerSizes, 8)

	var input []float32 = trainingData[len(trainingData)-1]
	n := initNetwork(layerSizes)

	for i := 0; i < 4980; i++ {
		learn(&n, trainingData, trainingLabels, 1, 4, len(trainingData)/4)
		if(i%10==0){
			fmt.Println("Training Cycle", i)
			validate(n, trainingData, trainingLabels, 8)
			validate(n, testData, testLabels, 8)
			fmt.Println()
		}
	}



	validate(n, testData, testLabels, 2)

	for i := 0; i < 120; i++ {
		var tmp []float32 = ask(&n,input);
		var min int = 0;
		for j := 1; j < len(tmp); j++ {
			if tmp[j]<tmp[min] { min = j }
		}
		var num int = 44
		for j := 0; j < num; j++ {
			input[12+j] = input[12+num+j]
		}
		for j := 0; j < num; j++ {
			input[12+num+j] = input[12+2*num+j]
		}
		for j := 0; j < num; j++ {
			input[12+2*num+j] = input[12+3*num+j]
		}
		input[12+3*num+min] = 1;
		fmt.Println(min)
		//fmt.Println(input)
		for j := 0; j < 12; j++ {
			if input[j]==1 {
				input[j] = 0
				if j==11 {
					input[0] = 1;
				} else {
					input[j+1] = 1;
				}
				break
			}
		}
	}
}

type netStruct struct {
	layerSizes      []int
	weights         [][][]float32
	biases          [][]float32
	completedCycles int
}
func initNetwork(layerSizes []int) netStruct {
	var n netStruct

	numLayers := len(layerSizes) - 1
	if numLayers < 1 {
		err := errors.New("Network needs at least two layers to function.")
		log.Fatal(err)
	}
	n.layerSizes = layerSizes
	n.weights = make([][][]float32, numLayers)
	n.biases = make([][]float32, numLayers)
	for i := 1; i < len(layerSizes); i++ {
		n.weights[i-1] = make([][]float32, layerSizes[i])
		for j := 0; j < len(n.weights[i-1]); j++ {
			n.weights[i-1][j] = make([]float32, layerSizes[i-1])
		}
		n.biases[i-1] = make([]float32, layerSizes[i])
	}

	for i := 0; i < len(n.weights); i++ {
		for j := 0; j < len(n.weights[i]); j++ {
			for k := 0; k < len(n.weights[i][j]); k++ {
				n.weights[i][j][k] = rand.Float32() - 0.5
			}
			n.biases[i][j] = rand.Float32() - 0.5
		}
	}

	return n
}

func squash(summation float32, bias float32) float32 {
	return 1.0 / float32((1.0 + math.Exp(float64((summation-bias)*-1.0))))
}

func fromBoolArray(array []bool) []float32 {
	output := make([]float32, len(array))
	for i := 0; i < len(array); i++ {
		if array[i] {
			output[i] = 0.9
		} else {
			output[i] = 0.1
		}

	}
	return output
}

func ask(network *netStruct, inputs []float32) []float32 {
	if len(inputs) != network.layerSizes[0] {
		err := errors.New("Network needs at least two layers to function.")
		log.Fatal(err)
	}

	previousLayerOutputs := inputs
	for i := 0; i < len(network.weights); i++ {
		layerOutputs := make([]float32, len(network.weights[i]))
		for j := 0; j < len(layerOutputs); j++ {
			summation := float32(0)
			for k := 0; k < len(network.weights[i][j]); k++ {
				summation += network.weights[i][j][k] * previousLayerOutputs[k]
			}
			layerOutputs[j] = squash(summation, network.biases[i][j])
		}
		previousLayerOutputs = layerOutputs
	}
	return previousLayerOutputs
}

func isCorrect(networkOutput []float32, label []bool) bool {
	index := 0
	for i := 0; i < len(networkOutput); i++ {
		if networkOutput[i] > networkOutput[index] {
			index = i
		}
	}
	return label[index]
}

func differenceArrays(networkOutput []float32, label []float32) float64 {
	total := float64(0)
	for i := 0; i < len(networkOutput); i++ {
		total += math.Pow(float64(networkOutput[i]-label[i]), 2)
	}
	return math.Sqrt(total)
}

func learn(network *netStruct, trainingData [][]float32, trainingLabels [][]bool, learningRate float32, numThreads int, rangePerThread int) {
	rangeStart := 0

	for rangeStart < len(trainingData)-1 {
		runnables := make([]*trainingRunnable, 0)
		var wg sync.WaitGroup
		for i := 0; i < numThreads; i++ {
			if rangeStart == len(trainingData)-1 {
				wg.Wait()
				continue
			}
			tmp := rangeStart + rangePerThread
			var rangeEnd int
			if tmp > len(trainingData) {
				rangeEnd = len(trainingData) - 1
			} else {
				rangeEnd = rangeStart + rangePerThread
			}
			wg.Add(1)
			thread := createTrainingRunnable(trainingData, trainingLabels, rangeStart, rangeEnd)
			runnables = append(runnables, &thread)
			go runTraining(&thread, network, &wg)
			rangeStart = rangeEnd
		}

		wg.Wait()

		var weightGradients [][][]float32
		var biasGradients [][]float32

		for i := 0; i < len(runnables); i++ {
			runnableGradients := runnables[i].weightGradients
			runnableBiasGradients := runnables[i].biasGradients
			if weightGradients == nil || biasGradients == nil {
				weightGradients = runnableGradients
				biasGradients = runnableBiasGradients
			} else {
				for i := 0; i < len(weightGradients); i++ {
					for j := 0; j < len(weightGradients[i]); j++ {
						for k := 0; k < len(weightGradients[i][j]); k++ {
							weightGradients[i][j][k] += runnableGradients[i][j][k]
						}
						biasGradients[i][j] += runnableBiasGradients[i][j]
					}
				}
			}
		}

		if weightGradients == nil || biasGradients == nil {
			err := errors.New("Gradient matrices were null.")
			log.Fatal(err)
		}
		updateInterval := numThreads * rangePerThread

		for i := 0; i < len(network.layerSizes)-1; i++ {
			for j := 0; j < network.layerSizes[i+1]; j++ {
				for k := 0; k < network.layerSizes[i]; k++ {
					network.weights[i][j][k] -= learningRate * weightGradients[i][j][k] / float32(updateInterval)
				}
				weightGradients[i][j] = make([]float32, network.layerSizes[i])
				network.biases[i][j] -= learningRate * biasGradients[i][j] / float32(updateInterval)
			}
			biasGradients[i] = make([]float32, network.layerSizes[i+1])
		}

	}

	network.completedCycles++
}

func validate(network netStruct, testData [][]float32, testLabels [][]bool, numThreads int) {
	fmt.Println("Starting validation (", network.completedCycles, "training cycles completed).")
	numCorrect := 0
	error := float64(0)

	runnables := make([]*validationRunnable, 0)
	var wg sync.WaitGroup
	rangeStart := 0
	for i := 0; i < numThreads; i++ {
		var rangeEnd int
		if i == numThreads-1 {
			rangeEnd = len(testData)
		} else {
			rangeEnd = (len(testData) / numThreads) + rangeStart
		}
		wg.Add(1)
		thread := createValidationRunnable(testData, testLabels, rangeStart, rangeEnd)
		runnables = append(runnables, &thread)
		go runValidation(&thread, &network, &wg)
		rangeStart = rangeEnd
	}

	wg.Wait()

	for i := 0; i < len(runnables); i++ {
		numCorrect += runnables[i].numCorrect
		error += runnables[i].totalError
	}
	error /= float64(len(testData))

	successRate := (numCorrect * 100) / len(testData)
	fmt.Println(": Network chose correctly in ", numCorrect, "/ ", len(testData), "cases (", successRate, "%) with an average error of ", error, " per input.")
}

type trainingRunnable struct {
	weightGradients [][][]float32
	biasGradients   [][]float32

	trainingData   [][]float32
	trainingLabels [][]bool
	rangeStart     int
	rangeEnd       int
}

func createTrainingRunnable(trainingData [][]float32, trainingLabels [][]bool, rangeStart int, rangeEnd int) trainingRunnable {
	var tr trainingRunnable
	tr.trainingData = trainingData
	tr.trainingLabels = trainingLabels
	tr.rangeStart = rangeStart
	tr.rangeEnd = rangeEnd
	return tr
}

func runTraining(tr *trainingRunnable, network *netStruct, wg *sync.WaitGroup) {
	numLayers := len(network.weights)
	outputs := make([][]float32, numLayers)
	errors := make([][]float32, numLayers)
	var previousLayerOutputs []float32
	tr.weightGradients = make([][][]float32, numLayers)
	tr.biasGradients = make([][]float32, numLayers)

	for x := 0; x < numLayers; x++ {
		tr.weightGradients[x] = make([][]float32, network.layerSizes[x+1])
		for i := 0; i < len(tr.weightGradients[x]); i++ {
			tr.weightGradients[x][i] = make([]float32, network.layerSizes[x])
		}
		tr.biasGradients[x] = make([]float32, network.layerSizes[x+1])
	}

	for setnumber := tr.rangeStart; setnumber < tr.rangeEnd; setnumber++ {
		input := tr.trainingData[setnumber]
		label := fromBoolArray(tr.trainingLabels[setnumber])

		previousLayerOutputs = input
		for i := 0; i < numLayers; i++ {
			currentLayerSize := network.layerSizes[i+1]
			outputs[i] = make([]float32, currentLayerSize)
			for j := 0; j < currentLayerSize; j++ {
				summation := float32(0)
				for k := 0; k < len(network.weights[i][j]); k++ {
					summation += network.weights[i][j][k] * previousLayerOutputs[k]
				}
				outputs[i][j] = squash(summation, network.biases[i][j])
			}
			previousLayerOutputs = outputs[i]
		}

		errors[len(errors)-1] = make([]float32, network.layerSizes[len(network.layerSizes)-1])
		for i := 0; i < len(errors[len(errors)-1]); i++ {
			actual := outputs[len(outputs)-1][i]
			error := actual - label[i]
			errors[len(errors)-1][i] = error * actual * (float32(1) - actual)
		}

		for i := numLayers - 2; i >= 0; i-- {
			errors[i] = make([]float32, network.layerSizes[i+1])
			for j := 0; j < len(errors[i]); j++ {
				error := float32(0)
				for k := 0; k < len(errors[i+1]); k++ {
					error += errors[i+1][k] * network.weights[i+1][k][j]
				}
				errors[i][j] = error * outputs[i][j] * (float32(1) - outputs[i][j])
			}
		}

		previousLayerOutputs = input
		for i := 0; i < numLayers; i++ {
			for j := 0; j < network.layerSizes[i+1]; j++ {
				for k := 0; k < len(previousLayerOutputs); k++ {
					tr.weightGradients[i][j][k] += errors[i][j] * previousLayerOutputs[k]
				}
				tr.biasGradients[i][j] += errors[i][j]
			}
			previousLayerOutputs = outputs[i]
		}
	}
	wg.Done()
}

type validationRunnable struct {
	numCorrect int
	totalError float64

	validationData   [][]float32
	validationLabels [][]bool
	rangeStart       int
	rangeEnd         int
}

func createValidationRunnable(validationData [][]float32, validationLabels [][]bool, rangeStart int, rangeEnd int) validationRunnable {
	var vr validationRunnable
	vr.validationData = validationData
	vr.validationLabels = validationLabels
	vr.rangeStart = rangeStart
	vr.rangeEnd = rangeEnd
	return vr
}

func runValidation(vr *validationRunnable, network *netStruct, wg *sync.WaitGroup) {
	for i := vr.rangeStart; i < vr.rangeEnd; i++ {
		output := ask(network, vr.validationData[i])
		usableLabel := fromBoolArray(vr.validationLabels[i])
		if isCorrect(output, vr.validationLabels[i]) {
			vr.numCorrect++
		}
		vr.totalError += differenceArrays(output, usableLabel)
	}
	wg.Done()
}

func initTrainingImages() [][]float32 {
	b, err := ioutil.ReadFile("trainingdata.json")
	//b, err := ioutil.ReadFile("TRAININGDATA.txt")
	if err != nil {
		fmt.Print(err)
	}
	var dat [][]float32
	if err := json.Unmarshal(b, &dat); err != nil {
		panic(err)
	}
	fmt.Println(len(dat))
	return dat
}
func initTrainingLabels() [][]bool {
	b, err := ioutil.ReadFile("traininglabels.json")
	//b, err := ioutil.ReadFile("TRAININGLABELS.txt")
	if err != nil {
		fmt.Print(err)
	}
	var dat [][]float32
	if err := json.Unmarshal(b, &dat); err != nil {
		panic(err)
	}
	fmt.Println(len(dat))
	labels := make([][]bool, len(dat))
	for i := range labels {
		labels[i] = make([]bool, len(dat[i]))
		for j := range labels[i] {
			if dat[i][j] == 1 {
				labels[i][j] = true
			} else {
				labels[i][j] = false
			}
		}
	}
	return labels
}
func initTestImages() [][]float32 {
	b, err := ioutil.ReadFile("trainingdata.json")
	//b, err := ioutil.ReadFile("VALIDDATA.txt")
	if err != nil {
		fmt.Print(err)
	}
	var dat [][]float32
	if err := json.Unmarshal(b, &dat); err != nil {
		panic(err)
	}
	fmt.Println(len(dat))
	return dat
}
func initTestLabels() [][]bool {
	b, err := ioutil.ReadFile("traininglabels.json")
	//b, err := ioutil.ReadFile("VALIDLABELS.txt")
	if err != nil {
		fmt.Print(err)
	}
	var dat [][]float32
	if err := json.Unmarshal(b, &dat); err != nil {
		panic(err)
	}
	fmt.Println(len(dat))
	labels := make([][]bool, len(dat))
	for i := range labels {
		labels[i] = make([]bool, len(dat[i]))
		for j := range labels[i] {
			if dat[i][j] == 1 {
				labels[i][j] = true
			} else {
				labels[i][j] = false
			}
		}
	}
	return labels
}
