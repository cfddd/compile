#include <bits/stdc++.h>
#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;
#define debug0(x) cout << "debug0: " << x << endl
#define fr(t, i, n) for (long long i = t; i < n; i++)
#define YES cout << "Yes" << endl
#define nO cout << "no" << endl
#define fi first
#define se second
#define int long long
using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef pair<int, int> PII;
typedef pair<LL, LL> PLL;

// #pragma GCC optimize(3,"Ofast","inline")
// #pragma GCC optimize(2)
int ask(int t)
{
    cout << "? " << t << endl;
    int res;
    cin >> res;
    return res;
}

void solve()
{
    int n;
    cin >> n;
    int l = 1, r = n;
    while(l+1 < r)
    {
        int mid = l + r >> 1;
        if(ask(mid))
            r = mid;
        else
            l = mid;
    }

    for (int i = l + 1; i <= r;i ++)if(ask(i))
            cout << "! " << i - 1 << endl;
}

signed main()
{
    /*
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    */
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
