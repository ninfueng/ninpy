#include <bits/stdc++.h>
using namespace std;
#define L(x) cout << x << endl

class Test {
    public:
        int a, b, c;
        Test(int aa, int bb, int cc): a(aa), b(bb), c(cc) {}
};

ostream& operator << (ostream& o, Test t) {
    o << t.a << " " << t.b << " " << t.c << endl;
}

int main() {
    Test test = Test(1, 2, 3);
    L(test.a << " " << test.b << " " << test.c);
    cout << test << endl;
}