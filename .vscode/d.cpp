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

void solve()
{
	string str;cin >> str;
	int n = str.size();
	map<char,int> cnt;
	long long ans = 0;
	for(int l = 0,r = 0;l < n;l ++)
	{
		while(r < n && str[r] != 'd')
		{
			r ++;
		}
		for(int ll = l,rr = l;ll < r;ll ++)
		{
			while(rr < r && (cnt['r'] == 0 || cnt['e'] == 0))
			{
				cnt[str[rr]]++;
				rr ++;
			}
			if(cnt['r'] && cnt['e'])ans += r - rr + 1;
			cnt[str[ll]] --;
			// debug2(rr,r);
		}
		l = r ++;
	}
	cout << ans << endl;
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
