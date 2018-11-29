#include<bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    string fileName;
    ofstream file;
    
    cout<<"Enter array size and filename"<<"\n";
    cin>>n>>fileName;

    file.open(fileName);


    file<<n<<"\n";

    for(int i=0; i<n; i++)
    {
        file<<rand()<<" ";
    }

    file.close();

    return 0;
}