// infobottleneck.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <cmath>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

#include <matplot/matplot.h>
#include <boost/range/combine.hpp>



namespace plt = matplot;

xt::xarray<double> DKL(xt::xarray<double> p, xt::xarray<double> q)
{

    /*
    Kulback - Leibler divergence D(A || B)
    :param A : pa(x), target distribution
    : param B : pb(x), other distribution
    : return : component - wise DKL(pa(x) || pb(x)), which is a tensor of the same dimensionality as A and B
    : each entry in the tensor is Ai* ln(Ai / Bi), which means the i - th component of DKL.
    : this code structure, to return the component - wise Dkl, rather than sum_i Ai* ln(Ai / Bi),
    : simplifies the update equations and the calculation of MI.
    : you will use it in the calculation of mutual information and in the update equations of IB.
  */

    xt::xarray<double> dkl = p * xt::log(p / q);

    // dkl[xt::nan_to_num(dkl)];

    // detect all nan in dkl and replace with 0

    for (auto& el : dkl)
    {
        if (std::isnan(el) || std::isinf(el))
        {
            el = 0.0;
        }
    }

    // std::cout << "dkl: " << dkl << "\n";

    return dkl;
}

// mutial information

xt::xarray<double> I(xt::xarray<double> pA, xt::xarray<double> pB, xt::xarray<double> pAB)
{

    /*
    mutual information I(X,Y) = DKL(P(x,y) || Px x Py)
    :param pA: A - p(a): marginal probability of X
    :param pB: B - p(b): marginal probability of Y
    :param pAB: p(a,b): joint probability of X and Y
    */

    xt::xarray<double> mi = DKL(pAB, pA * pB);
    xt::xarray<double> x = xt::sum(mi);

    // std::cout << "mi: " << x << "\n";
    return x;
}

// entropy

xt::xarray<double> H(xt::xarray<double> p)
{

    /*
        entropy H(X) = DKL(P(x) || 1)
            :param p: p(x): probability distribution of X
                */

    xt::xarray<double> h = DKL(p, 1);
    xt::xarray<double> x = (-1) * xt::sum(h);
    return x;
}

// make probs(tensor)

xt::xarray<double> make_probs(int dims1, int dims2, int dims3)
{

    xt::xarray<double> XY = xt::random::randn<double>({ dims1, dims2, dims3 });
    XY = XY - xt::amin(XY);
    XY = XY / xt::sum(XY);
    return XY;
}

// iterative information bottleneck

void IIB(int Xdim,
    int Ydim,
    int Mmax,
    int Mmin,
    int n_iters = 100,
    int n_tries = 3,
    int n_betas = 100,
    double beta_min = 0.1,
    double beta_max = 100)
{

    /*
                working with probabilities as tensors makes it easier to
                read/write code and lets us utilize tensor broadcasting.
                e.g, in this function we are working with discrete random
                variables X, Y, and T. We can represent all probabilities as
                tensors, where each dimension represents a random variable.
                                                                    */
                                                                    // e.g. p(x,y) is a tensor of shape (Xdim, Ydim, 1)
    xt::xarray<double> pXY = make_probs(Xdim, Ydim, 1);
    // for (auto& el : pXY.shape()) { std::cout << el << ", "; }
    // std::cout << "pXY: " << pXY.dimension()<<" "<<pXY << "\n";

    // p(x) is a tensor of shape (Xdim, 1, 1)
    xt::xarray<double> pX = xt::sum(pXY, { 1 }, xt::keep_dims);
    // for (auto& el : pX.shape()) { std::cout << el << ", "; }
    // std::cout << "pX: " << pX.dimension()<<" " <<  pX << "\n";

    // p(y) is a tensor of shape (1, Ydim, 1)
    xt::xarray<double> pY = xt::sum(pXY, { 0 }, xt::keep_dims);
    // for (auto& el : pY.shape()) { std::cout << el << ", "; }
    // std::cout << "pY: " << pY.dimension()<< " "<< pY << "\n";

    // entropy of pX
    xt::xarray<double> hX = H(pX);

    ////mutual information
    xt::xarray<double> target_MI = I(pX, pY, pXY);

    std::cout << "target MI: " << target_MI << "\n";

    // p(y | x) = p(x,y) / p(x)
    xt::xarray<double> pY_X = pXY / pX;

    // std::cout<<pY_X.dimension()<<" "<<pY_X<<"\n";

    // Lagrangian
    xt::xarray<double> Ls = xt::zeros<double>({ Mmax - Mmin + 1, n_betas });

    // relevance I(T;Y)
    xt::xarray<double> I_TYs = xt::zeros<double>({ Mmax - Mmin + 1, n_betas });

    // compression I(X;T)
    xt::xarray<double> I_TXs = xt::zeros<double>({ Mmax - Mmin + 1, n_betas });

    // betas
    xt::xarray<double> betas = xt::zeros<double>({ Mmax - Mmin + 1, n_betas });

    // for m, M in enumerate(range(Mmax, Mmin-1, -2)):
    // for i, beta in enumerate(np.linspace(beta_min, beta_max, n_betas)[:: - 1]) : convert to c++
    // np.linspace(beta_min, beta_max, n_betas)[::-1] = xt::linspace<double>(beta_min, beta_max, n_betas);
    // for i, beta in enumerate(np.linspace(beta_min, beta_max, n_betas)[:: - 1]) : convert to c++

    for (int m = 0, M = Mmax; M >= Mmin; m++, M -= 2)
    {
        std::cout << m << "\n";
        xt::xarray<double> betas = xt::linspace<double>(beta_min, beta_max, n_betas);
        // std::cout<<"betas: "<<betas<<"\n";
        xt::xarray<double> reversed_betas = xt::flip(betas, 0);
        // std::cout << "reversed_betas: " << reversed_betas << "\n";

        int i = 0;
        for (auto& beta : reversed_betas)
        {
            //std::cout << "beta: " << i << "\n";


            double L = INFINITY;
            double I_TX = NAN;
            double I_TY = NAN;

            // for _ in range(n_tries):
            for (int j = 0; j < n_tries; j++)
            {

                // initialize random p(t | x)
                xt::xarray<double> pT_X = make_probs(Xdim, 1, M);

                // std::cout << "pT_X: " << pT_X << "\n";

                // compute pT according to the Bayes rule  P_m(t) = sum_x(p(x) P_m(t | x))
                xt::xarray<double> pT = xt::sum(pX * pT_X, { 0, 1 }, xt::keep_dims);

                // std::cout << "pT: " << pT << "\n";

                // compute p(y | t)
                // P_m(y | t) = sum_x ( P_m(t | x) p(x,y) ) / P_m(t)

                xt::xarray<double> pY_T = xt::sum(pXY * pT_X / pT, { 0 }, xt::keep_dims);

                // std::cout << "pY_T: " << pY_T << "\n";

                for (int k = 0; k < n_iters; k++)
                {

                    xt::xarray<double> Z_X = xt::sum(pT * xt::exp((-1) * beta * xt::sum(DKL(pY_X, pY_T), { 1 }, xt::keep_dims)), { 2 }, xt::keep_dims);

                    // pT_X = pT_X * np.exp((-1)*beta*DKL(pY_X,pY_T)) / Z_X convert to c++
                    pT_X = (pT * xt::exp((-1) * beta * xt::sum(DKL(pY_X, pY_T), { 1 }, xt::keep_dims))) / Z_X;

                    // pT = np.sum(pX * pT_X, axis=(0,1), keepdims=True) convert to c++
                    pT = xt::sum(pX * pT_X, { 0, 1 }, xt::keep_dims);

                    // pY_T = np.sum(pXY * pT_X, axis=0) / pT convert to c++
                    pY_T = xt::sum(pXY * pT_X / pT, { 0 }, xt::keep_dims);

                    xt::xarray<double> I_TX_ = I(pT, pX, pT_X * pX);
                    //std::cout<<"I_TX_: "<<I_TX_<<"\n";

                    xt::xarray<double> I_TY_ = I(pT, pY, pY_T * pT);
                    //std::cout<<"I_TY_: "<<I_TY_<<"\n";

                    xt::xarray<double> L_ = I_TX_ - I_TY_ * beta;
                    //std::cout<<"L_: "<<L_<<"\n";

                    if (L_[0] < L)
                    {
                        L = L_[0];
                        I_TX = I_TX_[0];
                        I_TY = I_TY_[0];
                    }


                }

            }
            Ls[m, i] = L;
            //std::cout<<"Ls: "<<Ls<<"\n";
            I_TYs[m, i] = I_TY;
            //std::cout<<"I_TYs: "<<I_TYs<<"\n";
            I_TXs[m, i] = I_TX;
            //std::cout<<"I_XTs: "<<I_TXs<<"\n";
            betas[m, i] = beta;
            //std::cout<<"betas: "<<betas<<"\n";  
            i++;
        }
    }

    /*std::cout<<"Ls: "<<Ls<<"\n";
    std::cout<<"I_TYs: "<<I_TYs<<"\n";
    std::cout<<"I_XTs: "<<I_TXs<<"\n";*/


    std::vector<double> xv, yv;
    std::cout << "init plotting" << "\n";
    for (auto tup : boost::combine(I_TXs, I_TYs)) {

        double x, y;
        boost::tie(x, y) = tup;
        xv.push_back(x / hX[0]);
        yv.push_back(y / target_MI[0]);


        //std::cout << x << "," << y << "," << z << "\n";
    }


    plt::scatter(xv, yv);

    plt::title("Relevance-Compression Curves");
    plt::xlabel("I(T;X)/H(X)");
    plt::ylabel("(T;Y)/I(Y;X)");

    plt::show();




    /* plt::plot(axe);

     plt::show();*/
}

int main(int argc, char* argv[])
{
    //IIB(10, 5, 10, 1);
    IIB(10, 5, 10, 1, 100, 100, 1000, 0.1, 1000);
    return 0;
}
