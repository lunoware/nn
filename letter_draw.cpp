// letter_draw.cpp — interactive 5x7 grid letter recognition using the trained NN
// Build: g++ -o letter_draw letter_draw.cpp -lncurses -lm
// Run:   ./letter_draw

#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

// ---- NN (inference only) ----

struct Config {
    std::vector<int> layers;
};

struct JsonParser {
    const std::string& src;
    size_t pos;
    JsonParser(const std::string& s) : src(s), pos(0) {}

    void skipWS() {
        while (pos < src.size() && (src[pos]==' '||src[pos]=='\t'||src[pos]=='\n'||src[pos]=='\r'))
            pos++;
    }
    void expect(char c) {
        skipWS();
        if (pos >= src.size() || src[pos] != c) {
            fprintf(stderr, "JSON parse error: expected '%c'\n", c);
            exit(1);
        }
        pos++;
    }
    std::string parseString() {
        expect('"');
        std::string r;
        while (pos < src.size() && src[pos] != '"') r += src[pos++];
        expect('"');
        return r;
    }
    double parseNumber() {
        skipWS();
        char* end;
        double v = strtod(src.c_str() + pos, &end);
        pos = end - src.c_str();
        return v;
    }
    std::vector<int> parseIntArray() {
        std::vector<int> r;
        expect('['); skipWS();
        if (pos < src.size() && src[pos]==']') { pos++; return r; }
        r.push_back((int)parseNumber()); skipWS();
        while (pos < src.size() && src[pos]==',') { pos++; r.push_back((int)parseNumber()); skipWS(); }
        expect(']');
        return r;
    }
    std::vector<double> parseNumberArray() {
        std::vector<double> r;
        expect('['); skipWS();
        if (pos < src.size() && src[pos]==']') { pos++; return r; }
        r.push_back(parseNumber()); skipWS();
        while (pos < src.size() && src[pos]==',') { pos++; r.push_back(parseNumber()); skipWS(); }
        expect(']');
        return r;
    }
    std::vector<std::vector<double>> parseNestedArray() {
        std::vector<std::vector<double>> r;
        expect('['); skipWS();
        if (pos < src.size() && src[pos]==']') { pos++; return r; }
        r.push_back(parseNumberArray()); skipWS();
        while (pos < src.size() && src[pos]==',') { pos++; r.push_back(parseNumberArray()); skipWS(); }
        expect(']');
        return r;
    }
    std::vector<std::vector<std::vector<double>>> parseTripleNested() {
        std::vector<std::vector<std::vector<double>>> r;
        expect('['); skipWS();
        if (pos < src.size() && src[pos]==']') { pos++; return r; }
        r.push_back(parseNestedArray()); skipWS();
        while (pos < src.size() && src[pos]==',') { pos++; r.push_back(parseNestedArray()); skipWS(); }
        expect(']');
        return r;
    }
};

static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open '%s'\n", path.c_str()); exit(1); }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

struct Connection;
struct Neuron {
    std::vector<Connection*> conns;
    double value = 0, z = 0;
    bool isBias = false;
};
struct Connection { Neuron* src; double weight; };

struct Network {
    std::vector<std::unique_ptr<Neuron>>     neurons;
    std::vector<std::unique_ptr<Connection>> conns;
    std::vector<std::vector<Neuron*>>        layers;

    void build(const std::vector<int>& shape) {
        int N = (int)shape.size();
        layers.resize(N);

        auto addNeuron = [&](int li, bool bias) -> Neuron* {
            neurons.push_back(std::make_unique<Neuron>());
            neurons.back()->isBias = bias;
            if (bias) neurons.back()->value = 1.0;
            layers[li].push_back(neurons.back().get());
            return neurons.back().get();
        };

        for (int i = 0; i < shape[0]; i++) addNeuron(0, false);
        addNeuron(0, true);

        for (int li = 1; li < N; li++) {
            bool isOut = (li == N-1);
            for (int j = 0; j < shape[li]; j++) {
                Neuron* n = addNeuron(li, false);
                for (Neuron* src : layers[li-1]) {
                    conns.push_back(std::make_unique<Connection>());
                    conns.back()->src = src;
                    conns.back()->weight = 0;
                    n->conns.push_back(conns.back().get());
                }
            }
            if (!isOut) addNeuron(li, true);
        }
    }

    void loadWeights(const std::string& path) {
        std::string json = readFile(path);
        JsonParser p(json);
        p.expect('{'); p.skipWS(); p.parseString(); p.expect(':');
        auto w = p.parseTripleNested();
        for (int li = 1; li < (int)layers.size() && (li-1) < (int)w.size(); li++) {
            int ni = 0;
            for (Neuron* n : layers[li]) {
                if (n->isBias) continue;
                if (ni < (int)w[li-1].size())
                    for (int ci = 0; ci < (int)n->conns.size() && ci < (int)w[li-1][ni].size(); ci++)
                        n->conns[ci]->weight = w[li-1][ni][ci];
                ni++;
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        int ni = 0;
        for (Neuron* n : layers[0]) {
            if (!n->isBias && ni < (int)input.size()) n->value = input[ni++];
        }
        for (int li = 1; li < (int)layers.size(); li++) {
            for (Neuron* n : layers[li]) {
                if (n->isBias) continue;
                n->z = 0;
                for (Connection* c : n->conns) n->z += c->src->value * c->weight;
                n->value = 1.0 / (1.0 + exp(-n->z));
            }
        }
        std::vector<double> out;
        for (Neuron* n : layers.back()) if (!n->isBias) out.push_back(n->value);
        return out;
    }
};

// ---- UI ----

static const int GRID_COLS = 5;
static const int GRID_ROWS = 7;

// Color pair IDs
enum {
    CP_CELL_OFF = 1,
    CP_CELL_ON,
    CP_CURSOR_OFF,
    CP_CURSOR_ON,
    CP_HEADER,
    CP_LETTER,
    CP_BAR_HIGH,
    CP_BAR_LOW,
    CP_DIM,
    CP_BORDER,
};

static bool hasColors = false;

static void initColors() {
    start_color();
    use_default_colors();
    hasColors = (COLORS >= 8);
    if (!hasColors) return;

    // grid cells
    init_pair(CP_CELL_OFF,   COLOR_WHITE,   COLOR_BLACK);
    init_pair(CP_CELL_ON,    COLOR_WHITE,   COLOR_WHITE);
    init_pair(CP_CURSOR_OFF, COLOR_BLACK,   COLOR_CYAN);
    init_pair(CP_CURSOR_ON,  COLOR_BLACK,   COLOR_YELLOW);
    // header / letter display
    init_pair(CP_HEADER,     COLOR_CYAN,    -1);
    init_pair(CP_LETTER,     COLOR_YELLOW,  -1);
    // confidence bars
    init_pair(CP_BAR_HIGH,   COLOR_GREEN,   -1);
    init_pair(CP_BAR_LOW,    COLOR_WHITE,   -1);
    init_pair(CP_DIM,        COLOR_BLACK+8, -1);  // bright black = dark gray
    init_pair(CP_BORDER,     COLOR_CYAN,    -1);
}

// Draw a box around a region
static void drawBox(int y, int x, int h, int w) {
    if (hasColors) attron(COLOR_PAIR(CP_BORDER));
    mvaddch(y,     x,     ACS_ULCORNER);
    mvaddch(y,     x+w-1, ACS_URCORNER);
    mvaddch(y+h-1, x,     ACS_LLCORNER);
    mvaddch(y+h-1, x+w-1, ACS_LRCORNER);
    for (int i = 1; i < w-1; i++) {
        mvaddch(y,     x+i, ACS_HLINE);
        mvaddch(y+h-1, x+i, ACS_HLINE);
    }
    for (int i = 1; i < h-1; i++) {
        mvaddch(y+i, x,     ACS_VLINE);
        mvaddch(y+i, x+w-1, ACS_VLINE);
    }
    if (hasColors) attroff(COLOR_PAIR(CP_BORDER));
}

static void drawGrid(int startY, int startX, const bool grid[GRID_ROWS][GRID_COLS],
                     int curRow, int curCol) {
    // Each cell is displayed as 2 spaces wide, 1 char tall
    // Plus outer border: width = 2*GRID_COLS + 2, height = GRID_ROWS + 2
    int bw = 2 * GRID_COLS + 2;
    int bh = GRID_ROWS + 2;
    drawBox(startY, startX, bh, bw);

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            bool on  = grid[r][c];
            bool cur = (r == curRow && c == curCol);
            int pair;
            if (cur)
                pair = on ? CP_CURSOR_ON : CP_CURSOR_OFF;
            else
                pair = on ? CP_CELL_ON  : CP_CELL_OFF;

            if (hasColors) attron(COLOR_PAIR(pair));
            mvaddstr(startY + 1 + r, startX + 1 + c*2, "  ");
            if (hasColors) attroff(COLOR_PAIR(pair));
        }
    }
}

static void drawPredictions(int startY, int startX, const std::vector<double>& probs) {
    if (probs.empty()) return;

    // find argmax
    int best = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

    // Show big predicted letter
    if (hasColors) attron(COLOR_PAIR(CP_HEADER) | A_BOLD);
    mvprintw(startY, startX, "Prediction:");
    if (hasColors) attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);

    if (hasColors) attron(COLOR_PAIR(CP_LETTER) | A_BOLD);
    mvprintw(startY + 1, startX, "    %c   (%.1f%%)", 'A' + best, probs[best] * 100.0);
    if (hasColors) attroff(COLOR_PAIR(CP_LETTER) | A_BOLD);

    // Sort indices by probability descending for top-8 list
    std::vector<int> idx(26);
    for (int i = 0; i < 26; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    int barW = 14;
    if (hasColors) attron(COLOR_PAIR(CP_HEADER) | A_BOLD);
    mvprintw(startY + 3, startX, "Top letters:");
    if (hasColors) attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);

    for (int i = 0; i < 8 && i < 26; i++) {
        int li   = idx[i];
        double p = probs[li];
        int fill = (int)(p * barW + 0.5);

        int y = startY + 4 + i;

        if (hasColors) attron(li == best ? (COLOR_PAIR(CP_LETTER)|A_BOLD) : COLOR_PAIR(CP_DIM));
        mvprintw(y, startX, "%c", 'A' + li);
        if (hasColors) attroff(li == best ? (COLOR_PAIR(CP_LETTER)|A_BOLD) : COLOR_PAIR(CP_DIM));

        mvaddch(y, startX + 2, ' ');

        for (int b = 0; b < barW; b++) {
            if (b < fill) {
                if (hasColors) attron(li == best ? COLOR_PAIR(CP_BAR_HIGH) : COLOR_PAIR(CP_BAR_LOW));
                mvaddch(y, startX + 3 + b, ACS_CKBOARD);
                if (hasColors) attroff(li == best ? COLOR_PAIR(CP_BAR_HIGH) : COLOR_PAIR(CP_BAR_LOW));
            } else {
                if (hasColors) attron(COLOR_PAIR(CP_DIM));
                mvaddch(y, startX + 3 + b, ACS_CKBOARD);
                if (hasColors) attroff(COLOR_PAIR(CP_DIM));
            }
        }

        if (hasColors) attron(p > 0.5 ? COLOR_PAIR(CP_BAR_HIGH) : COLOR_PAIR(CP_DIM));
        mvprintw(y, startX + 3 + barW + 1, "%.0f%%", p * 100.0);
        if (hasColors) attroff(p > 0.5 ? COLOR_PAIR(CP_BAR_HIGH) : COLOR_PAIR(CP_DIM));
    }
}

static void drawHelp(int y, int x) {
    if (hasColors) attron(COLOR_PAIR(CP_DIM));
    mvprintw(y,   x, "Arrows: move   Space: toggle");
    mvprintw(y+1, x, "C: clear       Q: quit");
    if (hasColors) attroff(COLOR_PAIR(CP_DIM));
}

int main() {
    // Load NN
    std::string cfgJson = readFile("config.json");
    // parse layers array from config
    JsonParser cp(cfgJson);
    cp.expect('{'); cp.skipWS();
    std::vector<int> shape;
    while (cp.pos < cfgJson.size() && cfgJson[cp.pos] != '}') {
        std::string key = cp.parseString(); cp.expect(':');
        if (key == "layers") {
            shape = cp.parseIntArray();
        } else {
            // skip value
            cp.skipWS();
            while (cp.pos < cfgJson.size() && cfgJson[cp.pos] != ',' && cfgJson[cp.pos] != '}')
                cp.pos++;
        }
        cp.skipWS();
        if (cp.pos < cfgJson.size() && cfgJson[cp.pos] == ',') cp.pos++;
        cp.skipWS();
    }

    if (shape.empty() || shape[0] != GRID_ROWS * GRID_COLS) {
        fprintf(stderr, "Expected first layer size %d, got %d\n",
                GRID_ROWS * GRID_COLS, shape.empty() ? 0 : shape[0]);
        return 1;
    }

    Network net;
    net.build(shape);
    net.loadWeights("weights.json");

    // Init ncurses
    initscr();
    noecho();
    cbreak();
    keypad(stdscr, TRUE);
    curs_set(0);
    initColors();

    bool grid[GRID_ROWS][GRID_COLS] = {};
    int curRow = 0, curCol = 0;

    // Layout constants
    const int GRID_START_Y = 2;
    const int GRID_START_X = 2;
    const int PRED_START_X = GRID_START_X + 2*GRID_COLS + 4;
    const int PRED_START_Y = GRID_START_Y;
    const int HELP_Y       = GRID_START_Y + GRID_ROWS + 3;

    auto buildInput = [&]() {
        std::vector<double> inp;
        for (int r = 0; r < GRID_ROWS; r++)
            for (int c = 0; c < GRID_COLS; c++)
                inp.push_back(grid[r][c] ? 1.0 : 0.0);
        return inp;
    };

    std::vector<double> probs = net.predict(buildInput());

    bool running = true;
    while (running) {
        erase();

        // Title
        if (hasColors) attron(COLOR_PAIR(CP_HEADER) | A_BOLD);
        mvprintw(0, 2, " Letter Recognition ");
        if (hasColors) attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);

        drawGrid(GRID_START_Y, GRID_START_X, grid, curRow, curCol);
        drawPredictions(PRED_START_Y, PRED_START_X, probs);
        drawHelp(HELP_Y, GRID_START_X);

        refresh();

        int ch = getch();
        bool changed = false;

        switch (ch) {
            case KEY_UP:    curRow = (curRow - 1 + GRID_ROWS) % GRID_ROWS; break;
            case KEY_DOWN:  curRow = (curRow + 1) % GRID_ROWS;             break;
            case KEY_LEFT:  curCol = (curCol - 1 + GRID_COLS) % GRID_COLS; break;
            case KEY_RIGHT: curCol = (curCol + 1) % GRID_COLS;             break;
            case ' ':
                grid[curRow][curCol] = !grid[curRow][curCol];
                changed = true;
                break;
            case 'c': case 'C':
                memset(grid, 0, sizeof(grid));
                changed = true;
                break;
            case 'q': case 'Q':
                running = false;
                break;
        }

        if (changed)
            probs = net.predict(buildInput());
    }

    endwin();
    return 0;
}
