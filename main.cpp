#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
using namespace std;

typedef function<double(double)> func1;
typedef function<double(double, double)> func2;

typedef pair <double, const char *> valuePair;
typedef function<valuePair(double)> funcValue;

typedef vector<double> vect;
typedef vector<vect> matrix;

vector<funcValue> lambdaValue = vector<funcValue>({
	{ [](double u) { return valuePair(1, "1"); }},
	{ [](double u) { return valuePair(u, "u"); }},
	{ [](double u) { return valuePair(u * u, "u^2"); }},
	{ [](double u) { return valuePair(u * u * u, "u^3"); }},
	{ [](double u) { return valuePair(exp(u), "e^u"); }},
	{ [](double u) { return valuePair(cos(u), "cos(u)"); }},
	});
vector<func1> dldu = vector<func1>({
	{ [](double u) { return 0; }},
	{ [](double u) { return 1; }},
	{ [](double u) { return 2 * u; }},
	{ [](double u) { return 3 * u * u; }},
	{ [](double u) { return exp(u); }},
	{ [](double u) { return -sin(u); }},
	});

vector<func2> uValue = vector<func2>({
	{ [](double x, double t) { return x; } },
	{ [](double x, double t) { return t; } },
	{ [](double x, double t) { return x + t; } },
	{ [](double x, double t) { return x * x + t; } },
	{ [](double x, double t) { return x + t * t; } },
	{ [](double x, double t) { return exp(x) + exp(t); } },
	{ [](double x, double t) { return cos(x) + cos(t); } },
	});

int i, xDivKK, iterCount = -1;
double hx, ht, currRelDiscr = -1, currError = -1;

double sigma;
funcValue lambda;
func2  u, f;
matrix localA;
vect   di, al, au, b, bLocal, q, qPrev, qPast, times;
vector<vect> all_q;

int maxIter;
double h = 1e-5, w = 1, error, relDiscr;

struct gridParameters
{
	struct coordinateParameters
	{
		int count, elemCount;
		bool uniFlag;
		double x1, x2, qx;
	} coordinate;
	struct timeParameters
	{
		int count;
		bool uniFlag;
		double t1, t2, qt;
	} time;
} grid;

struct Node { double x; };
vector<Node> nodes;

double operator * (const vect &a, const vect &b)
{
	double tmp = 0;
	for (int i = 0; i < a.size(); i++) tmp += a[i] * b[i];
	return tmp;
}
vect operator - (const vect &a, const vect &b)
{
	vect tmp = a;
	for (int i = 0; i < b.size(); i++) tmp[i] -= b[i];
	return tmp;
}
vect multAByQ()
{
	int i = 0;
	vect tmp; tmp.resize(di.size());
	tmp[i] = di[i] * q[i] + au[i] * q[i + 1];
	for (i = 1; i < di.size() - 1; i++)
		tmp[i] = al[i - 1] * q[i - 1] + di[i] * q[i] + au[i] * q[i + 1];
	tmp[i] = al[i - 1] * q[i - 1] + di[i] * q[i];

	return tmp;
}

inline double norm(const vect &x)
{
	return sqrt(x * x);
}
inline bool isSolved()
{
	//return (currRelDiscr < relDiscr) || (currError < error) || (iterCount > maxIter);
	return (currRelDiscr < relDiscr) || (iterCount > maxIter);
}

double solve_error()
{
	double tmp = 0, tt = 0, uAnalitic, error;

	//for (int t = 0; t < grid.time.count; t++) {
	//	for (int i = 0; i < grid.coordinate.count; i++) {
	//		uAnalitic = u(nodes[i].x, times[t]);
	//		tmp += (all_q[t][i] - uAnalitic) * (all_q[t][i] - uAnalitic);
	//	}
	//}
	//error = sqrt(tmp) / (grid.time.count * grid.coordinate.count);

	tmp = tt = 0;
	for (int i = 0; i < grid.coordinate.count; i++) {
		uAnalitic = u(nodes[i].x, times[grid.time.count / 2]);
		tmp += (all_q[grid.time.count / 2][i] - uAnalitic) * (all_q[grid.time.count / 2][i] - uAnalitic);
		tt += uAnalitic * uAnalitic;
	}
	error = sqrt(tmp) / sqrt(tt);


	/*printf_s("t/2: %.2lf, x/2: %.2lf, q: %.5lf, u: %.5lf ///// ",
		times[grid.time.count / 2], nodes[grid.coordinate.count / 2].x,
		all_q[grid.time.count / 2][grid.coordinate.count / 2], u(nodes[grid.coordinate.count / 2].x, times[grid.time.count / 2]));*/
	error = fabs(all_q[grid.time.count / 2][grid.coordinate.count / 2] - u(nodes[grid.coordinate.count / 2].x, times[grid.time.count / 2]));

	return error;
}

void solveLU()
{
	// LU-factorization.
	for (int i = 1; i < di.size(); i++) {
		au[i - 1] = au[i - 1] / di[i - 1];
		di[i] = di[i] - al[i - 1] * au[i - 1];
	}
	// Forward Gauss.
	q[0] = b[0] / di[0];
	for (int i = 1; i < di.size(); i++)
		q[i] = (b[i] - al[i - 1] * q[i - 1]) / di[i];
	b = q;
	// Backward Gauss.
	int lidx = di.size() - 1;
	q[lidx] = b[lidx];
	for (int i = lidx - 1; i >= 0; i--)
		q[i] = (b[i] - au[i] * q[i + 1]);
}

void makeLocalA(int el, double t, double dt, bool isNewton)
{
	const double hx = nodes[el + 1].x - nodes[el].x;
	//double u1 = u(nodes[el].x, t), u2 = u(nodes[el + 1].x, t);
	double u1 = qPrev[el], u2 = qPrev[el + 1];

	// Local G matrix.
	double l0 = lambda(u1).first, l1 = lambda(u2).first;

	localA[0][0] = localA[1][1] = +(l0 + l1) / (2 * hx);
	localA[0][1] = localA[1][0] = -(l0 + l1) / (2 * hx);
	// Local M matrix.
	localA[0][0] = localA[1][1] += 2 * (sigma * hx) / (6 * dt);
	localA[0][1] = localA[1][0] += 1 * (sigma * hx) / (6 * dt);
	// Newton.
	if (isNewton) {
		localA[0][0] += (dldu[1](u1) * qPrev[el] - dldu[1](u1) * qPrev[el + 1]) / (2 * hx);
		localA[0][1] += (dldu[1](u2) * qPrev[el] - dldu[1](u2) * qPrev[el + 1]) / (2 * hx);
		localA[1][0] += (-dldu[1](u1) * qPrev[el] + dldu[1](u1) * qPrev[el + 1]) / (2 * hx);
		localA[1][1] += (-dldu[1](u2) * qPrev[el] + dldu[1](u2) * qPrev[el + 1]) / (2 * hx);
	}
}
void makeLocalb(int el, double t, double dt, bool isNewton)
{
	const double hx = nodes[el + 1].x - nodes[el].x;
	double u1 = u(nodes[el].x, t), u2 = u(nodes[el + 1].x, t);

	bLocal[0] = hx * (2 * f(nodes[el].x, t) + 1 * f(nodes[el + 1].x, t)) / 6 +
		sigma * hx * (2 * qPast[el] + 1 * qPast[el + 1]) / (6 * dt);

	bLocal[1] = hx * (1 * f(nodes[el + 1].x, t) + 2 * f(nodes[el + 1].x, t)) / 6 +
		sigma * hx * (1 * qPast[el] + 2 * qPast[el + 1]) / (6 * dt);

	if (isNewton) {
		bLocal[0] += (qPrev[el] - qPrev[el + 1]) * (dldu[1](u1) * qPrev[el] + dldu[1](u2) * qPrev[el + 1]) / (2 * hx);
		bLocal[1] -= (qPrev[el] - qPrev[el + 1]) * (dldu[1](u1) * qPrev[el] + dldu[1](u2) * qPrev[el + 1]) / (2 * hx);
	}
}
void makeGlobalA(double t, double dt, bool isNewton)
{
	const int xCount = grid.coordinate.count;
	di.clear(), au.clear(), al.clear();
	di.resize(xCount), al.resize(xCount - 1), au.resize(xCount - 1);
	for (int i = 0; i < xCount - 1; i++)
		di[i] = au[i] = al[i] = 0;
	di[xCount - 1] = 0;
	
	for (int el = 0; el < grid.coordinate.elemCount; el++) {
		makeLocalA(el, t, dt, isNewton);
		di[el] += localA[0][0];
		di[el + 1] += localA[1][1];
		au[el] += localA[0][1];
		al[el] += localA[1][0];
	}
	di[0] = di[xCount - 1] = 1;
	au[0] = al[al.size() - 1] = 0;
}
void makeGlobalb(double t, double dt, bool isNewton)
{
	const int xCount = grid.coordinate.count;
	b.clear(); b.resize(xCount);
	for (int i = 0; i < xCount; i++)
		b[i] = 0;

	for (int el = 0; el < grid.coordinate.elemCount; el++) {
		makeLocalb(el, t, dt, isNewton);
		b[el] += bLocal[0];
		b[el + 1] += bLocal[1];
	}
	b[0] = u(nodes[0].x, t);
	b[xCount - 1] = u(nodes[xCount - 1].x, t);
}

void debug()
{
	FILE *f;
	fopen_s(&f, "debug.txt", "w");

	for (int i = 0; i < grid.coordinate.count; i++)
		fprintf_s(f, "%5.2lf ", nodes[i].x);
	fprintf_s(f, "\n");
	for (int i = 0; i < grid.time.count; i++)
		fprintf_s(f, "%5.2lf ", times[i]);
	fprintf_s(f, "\n\n");

	for (int i = 0; i < di.size(); i++) {
		int j = 0;
		for (; j < i - 1; j++) fprintf_s(f, "%5s ", "0.00");
		if (i != 0) { fprintf_s(f, "%5.2lf ", al[i - 1]); j++; }
		fprintf_s(f, "%5.2lf ", di[i]); j++;
		if (i != di.size() - 1) { fprintf_s(f, "%5.2lf ", au[i]); j++; }
		for (; j < di.size(); j++) fprintf_s(f, "%5s ", "0.00");
		fprintf_s(f, "   %5.2lf\n", b[i]);
	}

	fprintf_s(f, "\n");
	for (int i = 0; i < di.size(); i++)
		fprintf_s(f, "%5.2lf ", q[i]);

	fprintf_s(f, "\n\n%5.2e, %5.2e", currRelDiscr, currError);
	fclose(f);
}

pair<int, double> solve(bool printDiscr = false, bool isNewton = false)
{
	all_q.resize(grid.time.count);
	for (int i = 0; i < all_q.size(); i++) all_q[i].resize(grid.coordinate.count);

	double t, dt;
	const int xCount = grid.coordinate.count;
	for (int i = 0; i < xCount; i++) all_q[0][i] = u(nodes[i].x, times[0]);
	qPast = qPrev = all_q[0];

	iterCount = 0;
	for (int i = 1; i < times.size(); i++) {
		t = times[i]; dt = t - times[i - 1];
		do {
			iterCount++;
			makeGlobalA(t, dt, isNewton), makeGlobalb(t, dt, isNewton);
			solveLU();
			//for (int j = 0; j < xCount; j++) q[j] = w * q[j] + (1 - w) * qPrev[j];
			currError = norm(q - qPrev) / norm(q);
			qPrev = q;
			makeGlobalA(t, dt, false), makeGlobalb(t, dt, false);
			currRelDiscr = norm(multAByQ() - b) / norm(b);
		} while (!isSolved());
		if (i == 1) debug();
		if (printDiscr) cout << "\r";
		all_q[i] = qPast = q;
	}
	return pair<int, double>(iterCount, solve_error());
}

void makeGridX(int xDivK)
{
	int &xCount = grid.coordinate.count;
	const double x1 = grid.coordinate.x1, x2 = grid.coordinate.x2;
	double &qx = grid.coordinate.qx;

	if (grid.coordinate.uniFlag) {
		hx = ((x2 - x1) / (xCount - 1)) / pow(2, xDivK);
		if (xDivK)
			xCount = (xCount - 1) * pow(2, xDivK) + 1;
	}
	else {
		int nx = xCount - 1;
		if (xDivK) {
			xCount = (xCount - 1) * pow(2, xDivK) + 1;
			nx *= pow(2, xDivK);
			qx *= pow(qx, 1.0 / xDivK);
		}
		hx = (x2 - x1) * (1 - qx) / (1 - pow(qx, nx));
	}
	grid.coordinate.elemCount = xCount - 1;
	nodes.resize(xCount);
	if (grid.coordinate.uniFlag) {
		nodes[0].x = x1;
		for (int i = 1; i < xCount - 1; i++)
			nodes[i].x = x1 + i * hx;
		nodes[xCount - 1].x = x2;
	}
	else {
		double dx = hx * qx, x = x1 + hx;
		nodes[0].x = x1;
		for (int i = 1; i < xCount - 1; i++, dx *= qx) {
			nodes[i].x = x;
			x += dx;
		}
		nodes[xCount - 1].x = x2;
	}

	localA.resize(2); localA[0].resize(2); localA[1].resize(2);
	di.resize(xCount), al.resize(xCount - 1), au.resize(xCount - 1);
	bLocal.resize(2); b.resize(xCount);
	q.resize(xCount), qPrev.resize(xCount);

	for (int i = 0; i < xCount - 1; i++)
		q[i] = qPrev[i] = b[i] = di[i] = al[i] = au[i] = 0;
	q[xCount - 1] = qPrev[xCount - 1] = b[xCount - 1] = di[xCount - 1] = 0;
	for (int i = 0; i < 2; i++) {
		bLocal[i] = 0;
		for (int j = 0; j < 2; j++)
			localA[i][j] = 0;
	}
}
void makeGridT(int tDivK)
{
	int &tCount = grid.time.count;
	const double t1 = grid.time.t1, t2 = grid.time.t2;
	double &qt = grid.time.qt;

	if (grid.time.uniFlag) {
		ht = ((t2 - t1) / (tCount - 1)) / pow(2, tDivK);
		if (tDivK)
			tCount = (tCount - 1) * pow(2, tDivK) + 1;
	}
	else {
		int nt = tCount - 1;
		if (tDivK) {
			tCount = (tCount - 1) * pow(2, tDivK) + 1;
			nt *= pow(2, tDivK);
			qt *= pow(qt, 1.0 / tDivK);
		}
		ht = (t2 - t1) * (1 - qt) / (1 - pow(qt, nt));
	}
	times.resize(tCount);
	if (grid.time.uniFlag)
	{
		times[0] = t1;
		for (int i = 1; i < tCount - 1; i++)
			times[i] = t1 + i * ht;
		times[tCount - 1] = t2;
	}
	else {
		double dt = ht * qt, t = t1 + ht;
		times[0] = t1;
		for (int i = 1; i < tCount - 1; i++, dt *= qt) {
			times[i] = t;
			t += dt;
		}
		times[tCount - 1] = t2;
	}
}
void input()
{
	int &xCount = grid.coordinate.count, &tCount = grid.time.count;
	bool &xUniFlag = grid.coordinate.uniFlag, &tUniFlag = grid.time.uniFlag;
	double &x1 = grid.coordinate.x1, &x2 = grid.coordinate.x2;
	double &t1 = grid.time.t1, &t2 = grid.time.t2;
	double &qx = grid.coordinate.qx, &qt = grid.time.qt;

	ifstream f("grid.txt");
	f >> xUniFlag >> tUniFlag;
	f >> x1 >> x2 >> xCount;
	if (!xUniFlag) f >> qx;
	f >> t1 >> t2 >> tCount;
	if (!tUniFlag) f >> qt;
	f.close();
}

func2 derivative(const func2 &f, const char *param)
{
	return [f, param](double x, double t) -> double {
		if (!strcmp(param, "x"))
			return (f(x, t) - f(x - h, t)) / h;
		else
			return (f(x, t) - f(x, t - h)) / h;
	};
}
func2 Lu()
{
	return [](double x, double t) -> double {
		func2 dudx = derivative(u, "x"), dudt = derivative(u, "t");
		func2 lambdaGrad = [dudx](double x, double t) -> double { return lambda(u(x, t)).first * dudx(x, t); };
		func2 div = derivative(lambdaGrad, "x");
		return -div(x, t) + sigma * dudt(x, t);
	};
}

int runFEM(int divCount = 1, const char *divParam = NULL, bool printDiscr = false, bool lambdaResearch = false, bool isNewton = false)
{
	if (divCount > 0) {
		if (strcmp(divParam, "x") && strcmp(divParam, "t")) {
			cout << "error: invalid param." << endl;
			return 1;
		}

		int xDivK, tDivK;

		cout << "Fragmentation of grid" << endl;
		cout << "parameter: " << divParam << ", fragmentation count: " << divCount << endl;

		cout << "x_nodes | t_nodes | iter | error norm\n" << scientific << setprecision(2);
		for (int divK = 0; divK < divCount; divK++)
		{
			input();
			if (divParam == NULL)            xDivK = 0, tDivK = 0;
			else if (!strcmp(divParam, "x")) xDivK = divK, tDivK = 0;
			else						     xDivK = 0, tDivK = divK;
			xDivKK = xDivK;
			makeGridX(xDivK), makeGridT(tDivK);
			const pair<int, double> result = solve(printDiscr, isNewton);
			const int xCount = grid.coordinate.count, tCount = grid.time.count;
			cout << setw(7) << xCount << " | " << setw(7) << tCount << " | " << setw(4) << result.first << " | " << result.second << endl;
		}
		cout << endl;
	}
	if (lambdaResearch) {
		cout << "Lambda research" << endl;
		cout << "lambda(u) | error norm\n" << scientific << setprecision(2);
		for (int i = 0; i < lambdaValue.size(); i++) {
			lambda = lambdaValue[i]; f = Lu(); input();
			makeGridX(NULL), makeGridT(NULL);
            double err_newton = solve(printDiscr, isNewton).second;
			cout << setw(9) << lambdaValue[i](1).second << " | " << err_newton << "\n";
		}
	}
	return 0;
}

int main()
{
	sigma = 1;
	u = uValue[4];
	lambda = lambdaValue[1];
	f = Lu();

	maxIter = 10000;
	error = relDiscr = 1e-11;
	
	const int divCount = 5;
	const char *divParam = "t";
	const bool printDiscr = true;
	const bool lambdaResearch = true;
	const bool isNewton = true;

	int res = runFEM(divCount, divParam, printDiscr, lambdaResearch, isNewton);
	return res;
}
//vector<func2> uValues = vector<func2>({
// [0] { [](double x, double t) { return x; } },
// [1] { [](double x, double t) { return t; } },
// [2] { [](double x, double t) { return x + t; } },
// [3] { [](double x, double t) { return x * x + t; } },
// [4] { [](double x, double t) { return x + t * t; } },
// [5] { [](double x, double t) { return exp(x) + exp(t); } },
// [6] { [](double x, double t) { return cos(x) + cos(t); } },
//	});
//vector<funcValue> lambdaValue = vector<funcValue>({
// [0] { [](double u) { return valuePair(1, "1"); }},
// [1] { [](double u) { return valuePair(u, "u"); }},
// [2] { [](double u) { return valuePair(u * u, "u^2"); }},
// [3] { [](double u) { return valuePair(u * u * u, "u^3"); }},
// [4] { [](double u) { return valuePair(exp(u), "e^u"); }},
// [5] { [](double u) { return valuePair(cos(u), "cos(u)"); }},
//	});