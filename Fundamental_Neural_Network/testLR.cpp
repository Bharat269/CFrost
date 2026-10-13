#include <iostream>
#include "sequential.h"
#include "Dense.h"
#include <vector>


//y= 3x1 ​− 2x2 ​+ 5
//test this eqn to see if the model learns i.e. the gradient descends

int main(){
    //std::cout<<"Start 1"<<std::endl;
    std::vector<tensor> x;
    std::vector<tensor> y;
    for(double i = -2;i<=2;i+=0.2){
        for(double j = -2; j<=2;j+=0.2){
            x.push_back({{2},{i,j}});
            y.push_back({{1},{3*i-2*j+5}});
        }
    }
    // std::cout<<"Start 2"<<std::endl;
    Seq model({2});
    model.add(std::make_unique<Dense>(2, "relu"));
    model.add(std::make_unique<Dense>(5, "relu"));
    model.add(std::make_unique<Dense>(5, "relu"));
    model.add(std::make_unique<Dense>(1, "linear"));
    
    model.compile();
    //std::cout<<"Start 3"<<std::endl;
    auto history = model.fit(x,y,100,0.001);
    //std::cout<<"Start 4"<<std::endl;
    for(auto &x:history) std::cout<<"Train error: "<<x.first<<"  Test error: "<<x.second<<std::endl;
    tensor test({2},{3,-3});
    auto res = model.pred(test);
    std::cout<<std::endl<<"Prediction for x1,x2 = 3,-3 is: "<<res.getValArray()[0]<<std::endl;

}