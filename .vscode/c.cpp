#include <bits/stdc++.h>

#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;
#define vec(a)                         \
    for (int i = 0; i < a.size(); i++) \
        cout << a[i] << ' ';           \
    cout << endl;
#define darr(a, _i, _n)               \
    cout << #a << ':';                \
    for (int ij = _i; ij <= _n; ij++) \
        cout << a[ij] << ' ';         \
    cout << endl;

#define endl "\n"

#define fi first
#define se second
#define caseT \
    int T;    \
    cin >> T; \
    while (T--)
// #define int long long
// #define int __int128

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef pair<int, int> PII;
typedef pair<LL, LL> PLL;

const double PI = acos(-1.0);

const int N = 1e6+10;

// 大整数加法函数，实现高精度加法
string add(const string& a, const string& b) {
    string result;
    int carry = 0;

    int maxLength = max(a.length(), b.length());

    for (int i = 1; i <= maxLength; ++i) {
        int digitA = (i <= a.length()) ? (a[a.length() - i] - '0') : 0;
        int digitB = (i <= b.length()) ? (b[b.length() - i] - '0') : 0;

        int sum = digitA + digitB + carry;

        carry = sum / 10;

        result.push_back('0' + sum % 10);
    }

    if (carry > 0) {
        result.push_back('0' + carry);
    }

    reverse(result.begin(), result.end());

    return result;
}

void solve()
{
	int d;
	string num;cin >> d >> num;
	
	int dotIndex = 0;
	for(;dotIndex < num.size();dotIndex ++)
	{
		if(num[dotIndex] == '.')break;
	}
	
	string noDot = num.substr(0,dotIndex) + num.substr(dotIndex+1);
	
	for (int i = 0; i < d; ++i) {
        noDot = add(noDot, noDot); 
    }
	
	int carry = 0;
	for(int i = noDot.size()-1;i >= noDot.size()-1-dotIndex;i --)
	{
		int cur = noDot[i] - '0' + carry;
		if(cur >= 5)
		{
			carry = 1;
		}else{
			carry = 0;
		}
	}
	noDot = noDot.substr(0,noDot.size() - 1 - dotIndex);
	
	if(carry)noDot = add(noDot,"1");
	
	cout << noDot << endl;
}

signed main()
{

    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    // caseT
    solve();

    return 0;
}
/*

*/
