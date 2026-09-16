#include <iostream>
#include "sequential.h"
#include <vector>
using namespace std;
//y=3x1​−2x2​+5
//test this eqn to see if the model learns i.e. the gradient descends

int main(){
    vector<vector<double>> x;
    vector<vector<double>> y;
    for(double i = -2;i<=2;i+=0.2){
        for(double j = -2; j<=2;j+=0.2){
            x.push_back({});
            y.push_back({});
            x[x.size()-1].push_back(i);
            x[x.size()-1].push_back(j);
            y[x.size()-1].push_back(3*i-2*j+5);
        }
    }
    Seq model(2, 5, "relu", "mse");
    model.add(5, "relu");
    model.add(1, "Linear");
    model.compile();
    auto history = model.fit(x,y,50,0.001);
    for(auto &x:history)cout<<"Train error: "<<x.first<<"  Test error: "<<x.second<<endl;
    auto res = model.pred({3.0,-3.0});
    cout<<"Prediction for x1,x2 = 3,-3 is: "<<res[0];
    int r;
    while(cin>>r);
}