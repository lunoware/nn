#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <memory>

// ---- Data Structs ----

struct Config {
    std::vector<int> layers;
    double learningRate;
    int epochs;
    int seed;
};

struct TrainingData {
    std::vector<std::vector<double>> inputs;   // [sample][feature]
    std::vector<std::vector<double>> targets;  // [sample][output]
};

// ---- Connection ----

class Connection {
public:
    class Neuron* neuron;
    double weight;

    Connection(class Neuron* n, double w) : neuron(n), weight(w) {}
};

// ---- Neuron ----

class Neuron {
public:
    std::vector<Connection*> connections;
    double value;
    double z;      // pre-activation weighted sum
    double delta;  // gradient accumulator
    std::string name;
    bool isBias;

    Neuron(double v, std::string n, bool bias = false)
        : value(v), z(0), delta(0), name(n), isBias(bias) {}

    void addConnection(Connection* c) {
        connections.push_back(c);
    }

    void calculate() {
        value = 0;
        for (auto* c : connections) {
            value += c->neuron->value * c->weight;
        }
        z = value;
    }

    void activation() {
        value = 1.0 / (1.0 + exp(-z));
    }

    double sigmoidDerivative() {
        return value * (1.0 - value);
    }

    void resetGradient() {
        delta = 0;
    }
};

// ---- Layer ----

class Layer {
public:
    std::vector<Neuron*> neurons;

    void addNeuron(Neuron* n) {
        neurons.push_back(n);
    }

    void updateNeurons() {
        for (auto* n : neurons) {
            if (n->isBias) continue;
            n->calculate();
            n->activation();
        }
    }

    void resetGradients() {
        for (auto* n : neurons) {
            if (n->isBias) continue;
            n->resetGradient();
        }
    }
};

// ---- Utility ----

double randWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open file '%s'\n", path.c_str());
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

// ---- JSON Parser ----

struct JsonParser {
    const std::string& src;
    size_t pos;

    JsonParser(const std::string& s) : src(s), pos(0) {}

    void skipWhitespace() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' ||
               src[pos] == '\n' || src[pos] == '\r'))
            pos++;
    }

    void expect(char c) {
        skipWhitespace();
        if (pos >= src.size() || src[pos] != c) {
            fprintf(stderr, "JSON parse error: expected '%c' at pos %zu, got '%c'\n",
                    c, pos, pos < src.size() ? src[pos] : '?');
            exit(1);
        }
        pos++;
    }

    std::string parseString() {
        expect('"');
        std::string result;
        while (pos < src.size() && src[pos] != '"') {
            result += src[pos++];
        }
        expect('"');
        return result;
    }

    double parseNumber() {
        skipWhitespace();
        char* end;
        double val = strtod(src.c_str() + pos, &end);
        pos = end - src.c_str();
        return val;
    }

    std::vector<int> parseIntArray() {
        std::vector<int> result;
        expect('[');
        skipWhitespace();
        if (pos < src.size() && src[pos] == ']') { pos++; return result; }
        result.push_back((int)parseNumber());
        skipWhitespace();
        while (pos < src.size() && src[pos] == ',') {
            pos++;
            result.push_back((int)parseNumber());
            skipWhitespace();
        }
        expect(']');
        return result;
    }

    std::vector<double> parseNumberArray() {
        std::vector<double> result;
        expect('[');
        skipWhitespace();
        if (pos < src.size() && src[pos] == ']') { pos++; return result; }
        result.push_back(parseNumber());
        skipWhitespace();
        while (pos < src.size() && src[pos] == ',') {
            pos++;
            result.push_back(parseNumber());
            skipWhitespace();
        }
        expect(']');
        return result;
    }

    std::vector<std::vector<double>> parseNestedArray() {
        std::vector<std::vector<double>> result;
        expect('[');
        skipWhitespace();
        if (pos < src.size() && src[pos] == ']') { pos++; return result; }
        result.push_back(parseNumberArray());
        skipWhitespace();
        while (pos < src.size() && src[pos] == ',') {
            pos++;
            result.push_back(parseNumberArray());
            skipWhitespace();
        }
        expect(']');
        return result;
    }

    // Triple nested: [[[w,w],[w,w]], [[w,w,w,w]]]
    std::vector<std::vector<std::vector<double>>> parseTripleNestedArray() {
        std::vector<std::vector<std::vector<double>>> result;
        expect('[');
        skipWhitespace();
        if (pos < src.size() && src[pos] == ']') { pos++; return result; }
        result.push_back(parseNestedArray());
        skipWhitespace();
        while (pos < src.size() && src[pos] == ',') {
            pos++;
            result.push_back(parseNestedArray());
            skipWhitespace();
        }
        expect(']');
        return result;
    }
};

Config parseConfig(const std::string& json) {
    Config cfg;
    JsonParser p(json);
    p.expect('{');
    p.skipWhitespace();
    while (p.pos < json.size() && json[p.pos] != '}') {
        std::string key = p.parseString();
        p.expect(':');
        if (key == "layers") {
            cfg.layers = p.parseIntArray();
        } else if (key == "learning_rate") {
            cfg.learningRate = p.parseNumber();
        } else if (key == "epochs") {
            cfg.epochs = (int)p.parseNumber();
        } else if (key == "seed") {
            cfg.seed = (int)p.parseNumber();
        } else {
            // skip unknown scalar value
            while (p.pos < json.size() && json[p.pos] != ',' && json[p.pos] != '}')
                p.pos++;
        }
        p.skipWhitespace();
        if (p.pos < json.size() && json[p.pos] == ',') p.pos++;
        p.skipWhitespace();
    }
    p.expect('}');
    return cfg;
}

TrainingData parseTrainingData(const std::string& json) {
    TrainingData td;
    JsonParser p(json);
    p.expect('{');
    p.skipWhitespace();
    while (p.pos < json.size() && json[p.pos] != '}') {
        std::string key = p.parseString();
        p.expect(':');
        if (key == "inputs") {
            td.inputs = p.parseNestedArray();
        } else if (key == "targets") {
            td.targets = p.parseNestedArray();
        } else {
            while (p.pos < json.size() && json[p.pos] != ',' && json[p.pos] != '}')
                p.pos++;
        }
        p.skipWhitespace();
        if (p.pos < json.size() && json[p.pos] == ',') p.pos++;
        p.skipWhitespace();
    }
    p.expect('}');
    return td;
}

// ---- CSV Parser ----

std::vector<std::vector<double>> parseCSV(const std::string& path) {
    std::vector<std::vector<double>> result;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open CSV '%s'\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        const char* ptr = line.c_str();
        while (*ptr) {
            char* end;
            double v = strtod(ptr, &end);
            if (end == ptr) break;
            row.push_back(v);
            ptr = end;
            if (*ptr == ',') ptr++;
        }
        if (!row.empty()) result.push_back(row);
    }
    return result;
}

// ---- Network ----

class Network {
    std::vector<std::unique_ptr<Neuron>>     allNeurons;
    std::vector<std::unique_ptr<Connection>> allConnections;
    std::vector<Layer>                        layers;
    double learningRate;

public:
    void build(const Config& cfg) {
        learningRate = cfg.learningRate;
        layers.clear();
        allNeurons.clear();
        allConnections.clear();

        int N = (int)cfg.layers.size();

        // Input layer: regular neurons + bias
        layers.push_back(Layer());
        for (int i = 0; i < cfg.layers[0]; i++) {
            allNeurons.push_back(std::make_unique<Neuron>(0.0, "i" + std::to_string(i)));
            layers[0].addNeuron(allNeurons.back().get());
        }
        allNeurons.push_back(std::make_unique<Neuron>(1.0, "b0", true));
        layers[0].addNeuron(allNeurons.back().get());

        // Hidden and output layers
        for (int li = 1; li < N; li++) {
            layers.push_back(Layer());
            bool isOutput = (li == N - 1);

            for (int j = 0; j < cfg.layers[li]; j++) {
                allNeurons.push_back(std::make_unique<Neuron>(0.0,
                    "n" + std::to_string(li) + "_" + std::to_string(j)));
                Neuron* neuron = allNeurons.back().get();

                // Connect to all neurons in previous layer (including its bias)
                for (Neuron* src : layers[li - 1].neurons) {
                    allConnections.push_back(std::make_unique<Connection>(src, randWeight()));
                    neuron->addConnection(allConnections.back().get());
                }
                layers[li].addNeuron(neuron);
            }

            // Append bias to non-output layers
            if (!isOutput) {
                allNeurons.push_back(std::make_unique<Neuron>(1.0,
                    "b" + std::to_string(li), true));
                layers[li].addNeuron(allNeurons.back().get());
            }
        }
    }

    void forward(const std::vector<double>& input) {
        // Set input neurons
        int ni = 0;
        for (Neuron* n : layers[0].neurons) {
            if (n->isBias) continue;
            if (ni < (int)input.size()) n->value = input[ni++];
        }
        // Propagate through remaining layers
        for (int li = 1; li < (int)layers.size(); li++) {
            layers[li].updateNeurons();
        }
    }

    void backward(const std::vector<double>& targets) {
        // Reset all gradients
        for (auto& n : allNeurons) n->resetGradient();

        // Seed output layer deltas: delta = -(target - output) * sigmoid'
        Layer& out = layers.back();
        int ti = 0;
        for (Neuron* n : out.neurons) {
            if (n->isBias) continue;
            n->delta = -(targets[ti] - n->value) * n->sigmoidDerivative();
            ti++;
        }

        // Backprop through all non-input layers (back to front)
        for (int li = (int)layers.size() - 1; li >= 1; li--) {
            for (Neuron* n : layers[li].neurons) {
                if (n->isBias) continue;
                if (li < (int)layers.size() - 1) {
                    // Hidden layer: apply chain rule
                    n->delta *= n->sigmoidDerivative();
                }
                for (Connection* c : n->connections) {
                    double oldW = c->weight;
                    c->neuron->delta += n->delta * oldW;
                    c->weight -= learningRate * n->delta * c->neuron->value;
                }
            }
        }
    }

    void train(const Config& cfg, const TrainingData& td) {
        int nSamples = (int)td.inputs.size();
        for (int epoch = 0; epoch <= cfg.epochs; epoch++) {
            double totalLoss = 0;
            for (int ex = 0; ex < nSamples; ex++) {
                forward(td.inputs[ex]);

                // Accumulate MSE
                Layer& out = layers.back();
                int ti = 0;
                for (Neuron* n : out.neurons) {
                    if (n->isBias) continue;
                    double diff = td.targets[ex][ti] - n->value;
                    totalLoss += 0.5 * diff * diff;
                    ti++;
                }

                backward(td.targets[ex]);
            }
            if (epoch % 1000 == 0) {
                printf("Epoch %d | MSE: %.6f\n", epoch, totalLoss / nSamples);
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        forward(input);
        std::vector<double> output;
        for (Neuron* n : layers.back().neurons) {
            if (!n->isBias) output.push_back(n->value);
        }
        return output;
    }

    void saveWeights(const std::string& path) {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) {
            fprintf(stderr, "Error: cannot write '%s'\n", path.c_str());
            exit(1);
        }
        fprintf(f, "{\"weights\":[\n");
        for (int li = 1; li < (int)layers.size(); li++) {
            fprintf(f, "  [");
            bool firstNeuron = true;
            for (Neuron* n : layers[li].neurons) {
                if (n->isBias) continue;
                if (!firstNeuron) fprintf(f, ",");
                firstNeuron = false;
                fprintf(f, "[");
                for (int ci = 0; ci < (int)n->connections.size(); ci++) {
                    if (ci > 0) fprintf(f, ",");
                    fprintf(f, "%.17g", n->connections[ci]->weight);
                }
                fprintf(f, "]");
            }
            fprintf(f, "]");
            if (li < (int)layers.size() - 1) fprintf(f, ",");
            fprintf(f, "\n");
        }
        fprintf(f, "]}\n");
        fclose(f);
    }

    void loadWeights(const std::string& path) {
        std::string json = readFile(path);
        JsonParser p(json);
        p.expect('{');
        p.skipWhitespace();
        std::string key = p.parseString();
        if (key != "weights") {
            fprintf(stderr, "Error: expected 'weights' key in %s\n", path.c_str());
            exit(1);
        }
        p.expect(':');
        auto weights = p.parseTripleNestedArray();

        // Restore weights in same construction order
        for (int li = 1; li < (int)layers.size() && (li - 1) < (int)weights.size(); li++) {
            int ni = 0;
            for (Neuron* n : layers[li].neurons) {
                if (n->isBias) continue;
                if (ni < (int)weights[li - 1].size()) {
                    for (int ci = 0; ci < (int)n->connections.size() &&
                                     ci < (int)weights[li - 1][ni].size(); ci++) {
                        n->connections[ci]->weight = weights[li - 1][ni][ci];
                    }
                }
                ni++;
            }
        }
    }
};

// ---- Main ----

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --train | --predict <inputs.csv>\n", argv[0]);
        return 1;
    }

    Config cfg = parseConfig(readFile("config.json"));
    srand(cfg.seed);

    Network net;

    if (strcmp(argv[1], "--train") == 0) {
        TrainingData td = parseTrainingData(readFile("data.json"));
        net.build(cfg);
        net.train(cfg, td);
        net.saveWeights("weights.json");
        printf("Weights saved to weights.json\n");

    } else if (strcmp(argv[1], "--predict") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s --predict <inputs.csv>\n", argv[0]);
            return 1;
        }
        net.build(cfg);
        net.loadWeights("weights.json");
        auto rows = parseCSV(argv[2]);
        for (auto& row : rows) {
            auto out = net.predict(row);
            for (int i = 0; i < (int)out.size(); i++) {
                if (i > 0) printf(",");
                printf("%.6f", out[i]);
            }
            printf("\n");
        }

    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
