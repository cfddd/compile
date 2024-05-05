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
int n,m,k;

long long acu(int l,int r)
{
	if(l > r)return 0;
	return 1LL*(l + r) * (r - l + 1) / 2;
}

bool check(int u)
{	
	long long sum = n;
	long long pre = acu(max(0,u - k + 1 - 1),u - 1);
	long long suf = acu(max(0,u - (n - k) - 1),u-2);
	return pre + suf + sum > m;
}

void solve()
{
	cin >> n >> m >> k;
	
	int l = 1,r = m;
	while(l + 1 < r){
		int mid = l + r >> 1;
		if(check(mid))r = mid;
		else l = mid;
	}
	if(!check(r))cout << r << endl;
	else cout << l << endl;
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
