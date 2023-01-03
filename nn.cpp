#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <math.h>

class Neuron;

const double learningRate = 0.1;

/*
    Connection
    Connection from neuron to neuron, connection must be from child layer to parent layer
*/
class Connection {
    public: 
        Neuron* neuron;
        double weight;

        Connection(Neuron* n, double w) {
            neuron = n;
            weight = w;
        }
};

/*
    Neuron 
    Gets signal from Neurons in previous layer and calculates its value for next layer calculations  
*/
class Neuron {
	public: 
        std::vector<Connection*> connections;
        double value;
        std::string name;

        Neuron(double v, std::string n) {
            value = v;
            name = n;

        }

        /*void connectToNeuron(Neuron* n, double w) {
            Connection c(n, w);
            connections.push_back(&c);
        }*/

        void addConnection(Connection* c){
            connections.push_back(c);
        }

        void calculate() {
            for (int i = 0; i < connections.size(); i++) {
                std::string pN = connections[i]->neuron->name;
                double cV = connections[i]->neuron->value;
                double cW = connections[i]->weight;
                double v = cV * cW;
                std::cout << pN << "->" << name << " " << cV << " " << cW << " " << v << std::endl;	
                value += v;
            }
        }

        void activation(){
            value = 1/(1+exp(-value));
        }


};

/* 
    Layer 
    Group neurons so they can are doing calculations in the right sequence.
*/ 
class Layer {
    public: 
        std::vector<Neuron*> neurons;
        void addNeuron(Neuron* n) {
            neurons.push_back(n);
        }
        void updateNeurons() {
            for (int i = 0; i < neurons.size(); i++) {
                neurons[i]->calculate();
                neurons[i]->activation();
            }
        }
        void backPropagate(double loss) {
            for (int i = 0; i < neurons.size(); i++) {
                for (int u = 0; u < neurons[i]->connections.size(); u++) {
                    double weight = neurons[i]->connections[u]->weight;

		     // Let’s calculate those deltas
		    double delta = weight * loss * 1;

		    // Updating the weights
		    double newWeight = weight - learningRate * neurons[i]->connections[u]->neuron->value * delta;

                    neurons[i]->connections[u]->weight = newWeight;
                }
            }
        }
};

int main() {
    Layer l1;
    Layer l2;
    Layer l3;

        //          L1     L2  L3

        //          x1 --- a1
        //            \   /  \
        //             \ /    \
        //              /      b0
        //             / \    /
        //            /   \  /
        //          x2 --- a2


        Neuron x1(0, "x1");
        Neuron x2(2, "x2");

        // Layer 1 
        l1.addNeuron(&x1);
        l1.addNeuron(&x2);

        Neuron a1(0, "a1");
        Connection a1x1(&x1, 1);
        a1.addConnection(&a1x1);
        Connection a1x2(&x2, 1);
        a1.addConnection(&a1x2);

        Neuron a2(0, "a2");
        Connection a2x1(&x1, 0);
        a2.addConnection(&a2x1);
        Connection a2x2(&x2, 1);
        a2.addConnection(&a2x2);

        // Layer 2
        l2.addNeuron(&a1);
        l2.addNeuron(&a2);


        Neuron d0(0, "d0");
        Connection d0a1(&a1, 0);
        d0.addConnection(&d0a1);
        Connection d0a2(&a2, 1);
        d0.addConnection(&d0a2);

        l3.addNeuron(&d0);


        
    l2.updateNeurons();
    l3.updateNeurons();

    // Loss = actual_y - predicted_y

    double expectedVal = 0.5;

    double loss = expectedVal - d0.value;

    l3.backPropagate(loss);

    /*
        delta_D0 = total_loss = -4
        delta_Z0 = W . delta_D0 . f'(Z0) = 1 . (-4) . 1 = -4
        delta_Z1 = W . delta_D0 . f'(Z1) = 1 . (-4) . 1 = -4
        delta_Z2 = W . delta_D0 . f'(Z2) = 1 . (-4) . 1 = -4
        delta_Z3 = W . delta_D0 . f'(Z3) = 1 . (-4) . 1 = -4
    */
    l2.backPropagate(loss);

	std::cout << d0.value << std::endl;	
    
	return 0;
}
