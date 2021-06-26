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

void view(string_view test) {
    cout << test << '\n';
}

class Vector {
    public:
        Vector() = default;
        Vector(int num): data(new int[num]), size(num) {}
        ~Vector() { delete[] data; }

    private:
        int* data;
        int size;

};


int main() {
    // Test test = Test(1, 2, 3);
    // L(test.a << " " << test.b << " " << test.c);
    // cout << test << endl;

    vector<vector<int>> a;
    a = {{1,2}};
    L(a[0][0]);

    string testString = "Testing one two three";
    view(testString);
}