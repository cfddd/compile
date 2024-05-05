#include <bits/stdc++.h>
using namespace std;


#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;
// #define int long long
#define vec(a)                         \
    for (int i = 0; i < a.size(); i++) \
        cout << a[i] << ' ';           \
    cout << endl;
#define darr(a, _i, _n)               \
    cout << #a << ':';                \
    for (int ij = _i; ij <= _n; ij++) \
        cout << a[ij] << ' ';         \
    cout << endl;
#define fi first
#define se second



Solution solve;
int main()
{
	string s = "abcabc";
	vector<vector<int>> q = {{1,1,3,5},{0,2,5,5}};
	
	// string s = "abcd",target = "acbe";
	// vector<string> ori = {"a","b","c","c","e","d"};
	// vector<string> ch = {"b","c","b","e","b","e"};
	// vector<int> co = {2,5,5,1,2,20};
	
	// string s = "abcdefgh",target = "acdeeghh";
	// vector<string> ori = {"bcd","fgh","thh"};
	// vector<string> ch = {"cde","thh","ghh"};
	// vector<int> co = {1,3,5};
	
	// string s = "ajhpd",target = "djjdc";
	// vector<string> ori = {"hpd","iyk","qzd","hpi","aic","znh","cea","fug","wir","kwu","yjo","rzi","a","n","f","q","u","w","x","i","x","s","o","u"};
	// vector<string> ch = {"iyk","qzd","hpi","aic","znh","cea","fug","wir","kwu","yjo","rzi","jdc","n","f","q","u","w","x","i","x","s","o","u","d"};
	// vector<int> co = {80257,95140,96349,89449,81714,5859,96734,96109,41211,99975,57611,32644,82896,22164,99889,98061,95403,90922,64031,94558,58418,99717,96588,88286};
	
	cout << solve.canMakePalindromeQueries(s,q) << endl;
	
	
}
