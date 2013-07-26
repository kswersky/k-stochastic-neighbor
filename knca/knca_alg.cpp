/*
Code provided by Danny Tarlow, Kevin Swersky, Laurent Charlin, Ilya Sutskever and Rich Zemel

Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our 
web page.

The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This code implements the methods described in the paper:
Stochastic k-neighborhood selection for supervised and unsupervised learning. ICML 2013.
*/

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cmath>

#define DEBUG_PTRS 0
#define DEBUG_F1 0
#define DEBUG_F2 0
#define DEBUG_B2 0
#define DEBUG_B1 0

using namespace std;

void print_msgs(int D, int Kmax, double *msgs) {
  int K = Kmax+1;

  for (int d = 0; d < D; d++) {
    int dK = d*K;
    for (int k = 0; k < K; k++) {
      cout << msgs[dK+k] << " ";
    }
    cout << endl;
  }
}

inline double normalize(double *arr, int K) {
  double sum = 0;
  for (int k = 0; k < K; k++)  sum += arr[k];
  for (int k = 0; k < K; k++)  arr[k] /= sum;
  return log(sum);
}

double l1_forward(double *exp_node_pots, double *fmsgs, int K, int Nc, int Kp) {

  fmsgs[0] = 1;  fmsgs[1] = exp_node_pots[0]; 
  for (int k = 2; k <= K; k++) fmsgs[k] = 0;

  double log_Z = 0;
  for (int n = 0; n < Nc-1; n++) {
    int nK = n*(K+1);
    if (DEBUG_PTRS) {    
      if (n == Nc-2) {
        cout << "Last forward idx: " << &(fmsgs[nK+(K+1)]) << endl;
      }
    }

    for (int k = 0; k <= K; k++) {
      fmsgs[nK+(K+1)+k] = fmsgs[nK+k] + exp_node_pots[n+1] * ((k==0) ? 0 : fmsgs[nK+k-1]);
    }
    log_Z += normalize(&(fmsgs[nK+(K+1)]), K+1);
  }
  if (DEBUG_F1)  print_msgs(Nc, K, fmsgs);

  return log_Z;
}

double l2_forward(double *inA, double *inB, double *out, int K) {

  if (inB == NULL) {
    for (int k = 0; k <= K; k++)  out[k] = inA[k];

    if (DEBUG_F2) {
      cout << "l2 forward" << endl;    
      print_msgs(1, K, inA);
      print_msgs(1, K, out);
      cout << endl;
    }
    return 0;
  }
  
  for (int k = 0; k <= K; k++)  out[k] = 0;
  
  for (int kA = 0; kA <= K; kA++) {
    for (int kB = 0; kB <= K-kA; kB++) {
      // out=kA+kB
      out[kA+kB] += inA[kA]*inB[kB];
    }
  }
  if (DEBUG_F2) {  
    cout << "l2 forward" << endl;
    print_msgs(1, K, inA);
    print_msgs(1, K, inB);
    print_msgs(1, K, out);
    cout << endl;
  }
  return normalize(out, K+1);
}


void l2_backward(double *in1, double *in2, double *inpa, double *out1, double *out2, int K) {

  if (in2 == NULL) {
    for (int k = 0; k <= K; k++) out1[k] = inpa[k];
    if (DEBUG_B2) {
      cout << "l2 back" << endl;      
      cout << "inpa "; print_msgs(1, K, inpa);
      cout << "in1  "; print_msgs(1, K, in1);
      cout << "-->" << endl;
      cout << "out1 "; print_msgs(1, K, out1);
      //print_msgs(1, K, in2);
      cout << endl;
    }
    return;
  }
  for (int k = 0; k <= K; k++) {
    out1[k] = 0;
    out2[k] = 0;
  }
  for (int kpa = 0; kpa <= K; kpa++) {
    for (int kch = 0; kch <= kpa; kch++) {
      // kchi=kpa-kchj
      out1[kpa-kch] += in2[kch]*inpa[kpa];
      out2[kpa-kch] += in1[kch]*inpa[kpa];
      //cout << kpa-kch << " " << kch << " " << kpa << "  " << out1[kpa-kch] << endl;
    }
  }
  
  if (DEBUG_B2) {
    cout << "l2 back" << endl;
    cout << "inpa ";  print_msgs(1, K, inpa);
    cout << "in1  ";  print_msgs(1, K, in1);
    cout << "in2  ";  print_msgs(1, K, in2);
    cout << "-->" << endl;
    cout << "out1 ";  print_msgs(1, K, out1);
    cout << "out2 ";  print_msgs(1, K, out2);
    cout << endl;
  }
  
  normalize(out1, K+1);
  normalize(out2, K+1);
}


void l1_backward(double *exp_node_pots, double *bmsgs, int K, int Nc, int Kp) {
  for (int n = Nc-1; n > 0; n--) {
    int nK = n*(K+1);
    double sum = 0;
    if (DEBUG_PTRS) {
      if (n == Nc-1) {
        cout << "l1 back start: " << &(bmsgs[nK]) << endl;
      } else if (n == 0) {
        cout << "l1 back end: " << &(bmsgs[nK]) << endl;
      } else {
        cout << &(bmsgs[nK-(K+1)]) << endl;
      }
    }
    if (DEBUG_B1) {
      if (n == Nc-1) {
        cout << "first b1 ";  print_msgs(1, K, bmsgs + nK);
      }
    }
    
    for (int k = 0; k <= K; k++) {
      bmsgs[nK-(K+1)+k] = bmsgs[nK+k] + exp_node_pots[n] * ((k==K) ? 0 : bmsgs[nK+k+1]);
    }
    normalize(&(bmsgs[nK-(K+1)]), K+1);
  }
  if (DEBUG_B1) {
    cout << "b1 messages" << endl;
    print_msgs(Nc, K, bmsgs);
    cout << endl;
  }
}

extern "C" {
  void infer(int N, int B, int K, int Kp, double *class_counts,
             int C, double *exp_node_pots, double *ys,
             double *fmsgs1, double *bmsgs1, double *fmsgs2, double *bmsgs2,
             double *result_margs, double *result_log_Zs) {

    //cerr << "CLASS COUNTS.  CHECK THESE!!!" << endl;
    for (int c = 0; c < C; c++) {
      int Nc = static_cast<int>(class_counts[c]);
      //cerr << Nc << " ";
    }
    //cerr << endl;
    //cerr << "CLASS COUNTS.  CHECK THOSE!!!" << endl;
    //cerr << endl;
    
    for (int i = 0; i < N*(K+1); i++) {
      fmsgs1[i] = 0;
      bmsgs1[i] = 0;
    }
    
    for (int b = 0; b < B; b++) {
      double *exp_node_pots_b = exp_node_pots + b*N;
      int cum_count = 0;
      double log_Z = 0;

      // FORWARDS LEVEL 1
      for (int c = 0; c < C; c++) {
        int Nc = static_cast<int>(class_counts[c]);
        double *exp_node_pots_bc = exp_node_pots_b + cum_count;  // for batch and class
        double *fmsgs1_c = fmsgs1 + cum_count * (K+1);           // f1_offset(n,c,N,K,C)

        if (DEBUG_F1)  cout << "L1 " << c << endl;
        log_Z += l1_forward(exp_node_pots_bc, fmsgs1_c, K, Nc, Kp);
        cum_count += Nc;
      }

      // FORWARDS LEVEL 2
      cum_count = 0;
      for (int c = 0; c < C; c++) {
        int Nc = static_cast<int>(class_counts[c]);
        double *f1_in = fmsgs1 + (cum_count+Nc-1) * (K+1);  
        double *f2_in = (c > 0) ? fmsgs2 + c * (K+1) : NULL;
        double *f2_out = fmsgs2 + (c+1) * (K+1);

        if (DEBUG_PTRS) {        
          cout << "f2" << c << endl;
          cout << "f1_in:  " << f1_in << endl;
          cout << "f2_in:  " << f2_in << endl;
          cout << "f2_out: " << f2_out << endl;
          cout << endl;
        }
        
        bool is_true_class = (c == static_cast<int>(ys[b]));
        // SPECIAL OPTIONS:
        // Kp = -1: no per-class cardinalities
        // Kp = -2: true class >= int(K+1/2); others < int(K+1/2)
        if (Kp == -1) {
          // pass
        } else if (Kp == -2) {
          int thresh = (K+1)/2;  // intentional integer division
          if (is_true_class) {  // apply cardinality potential == Kp
            for (int k = 0; k <= K; k++)  f1_in[k] = (k >= thresh) * f1_in[k];
          } else {              // apply cardinality potential < Kp
            for (int k = 0; k <= K; k++)  f1_in[k] = (k < thresh) * f1_in[k];
          }          
        } else {  // normal behavior
          if (is_true_class) {  // apply cardinality potential == Kp
            for (int k = 0; k <= K; k++)  f1_in[k] = (k == Kp) * f1_in[k];
          } else {              // apply cardinality potential < Kp
            for (int k = 0; k <= K; k++)  f1_in[k] = (k < Kp) * f1_in[k];
          }
        }

        log_Z += l2_forward(f1_in, f2_in, f2_out, K);
        cum_count += Nc;
      }

      /******* DONE FORWARD PASS *********/
      
      double *root_msg = fmsgs2 + (C)*(K+1);
      if (DEBUG_PTRS) {
        cout << "root:  " << root_msg << endl;
      }
      
      // Can compute log_Z now
      log_Z += log(root_msg[K]);

      result_log_Zs[b] = log_Z;

      /******* START BACKWARD PASS *********/

      double *first_b2_msg = bmsgs2 + (C)*(K+1);
      for (int k = 0; k < K; k++)  first_b2_msg[k] = 0;  first_b2_msg[K] = 1;

      if (DEBUG_PTRS) {      
        cout << "1st b2: " << first_b2_msg << endl;
        cout << endl;
      }

      // BACKWARDS LEVEL 2
      // DO NOT DO cum_count = 0.  Want to start at end.
      for (int c = C-1; c >= 0; c--) {
        int Nc = static_cast<int>(class_counts[c]);
        cum_count -= Nc;

        double *f1_in  = fmsgs1 + (cum_count+Nc-1) * (K+1);   // comes from f1
        double *b1_out = bmsgs1 + (cum_count+Nc-1) * (K+1);

        double *bpa_in = bmsgs2 + (c+1) * (K+1);              // comes from b2
        double *f2_in  = (c > 0) ? fmsgs2 + c * (K+1) : NULL; // comes from f2
        double *b2_out = (c > 0) ? bmsgs2 + (c) * (K+1) : NULL;

        if (DEBUG_PTRS) {
          cout << "b2 " << c << endl;
          cout << "f1_in: " << f1_in << endl;
          cout << "f2_in: " << f2_in << endl;
          cout << "bpa_in:" << bpa_in << endl;
          cout << "b1_out:" << b1_out << endl;
          cout << "b2_out:" << b2_out << endl;
          cout << endl;
        }
        
        l2_backward(f1_in, f2_in, bpa_in, b1_out, b2_out, K);

        // Multiply in cardinality potential for backward pass
        bool is_true_class = (c == static_cast<int>(ys[b]));
        // SPECIAL OPTIONS:
        // Kp = -1: no per-class cardinalities
        // Kp = -2: true class >= int(K+1/2); others < int(K+1/2)
        if (Kp == -1) {
          // pass
        } else if (Kp == -2) {
          int thresh = (K+1)/2;  // intentional integer division
          if (is_true_class) {  // apply cardinality potential == Kp
            for (int k = 0; k <= K; k++)  b1_out[k] = (k >= thresh) * b1_out[k];
          } else {              // apply cardinality potential < Kp
            for (int k = 0; k <= K; k++)  b1_out[k] = (k < thresh) * b1_out[k];
          }          
        } else {  // normal behavior
          if (is_true_class) {  // apply cardinality potential == Kp
            for (int k = 0; k <= K; k++)  b1_out[k] = (k == Kp) * b1_out[k];
          } else {  // apply cardinality potential < Kp
            for (int k = 0; k <= K; k++)  b1_out[k] = (k < Kp) * b1_out[k];
          }
        }
        normalize(b1_out, K+1);
      }
      
      // BACKWARDS LEVEL 1
      cum_count = 0;
      for (int c = 0; c < C; c++) {
        int Nc = static_cast<int>(class_counts[c]);
        double *exp_node_pots_bc = exp_node_pots_b + cum_count;  // for batch and class
        //double *bmsg = bmsgs1 + (cum_count+Nc-1) * (K+1);
        double *bmsg = bmsgs1 + (cum_count) * (K+1);
        
        l1_backward(exp_node_pots_bc, bmsg, K, Nc, Kp);
        cum_count += Nc;
      }

      // COMPUTE BELIEFS
      double b0;
      double b1;
      int bN = b*N;
      cum_count = 0;
      for (int c = 0; c < C; c++) {
        int Nc = static_cast<int>(class_counts[c]);
        
        for (int n = 0; n < Nc; n++) {
          int nn = cum_count + n;
          int nK = nn*(K+1);
          b0 = 0; b1 = 0;

          //for (int k = 0; k <= K; k++)  cout << bmsgs1[nK+k]*fmsgs1[nK+k] << " ";
          //cout << endl;
          if (n > 0) {
            for (int k = 0; k <= K; k++) b0 += bmsgs1[nK+k] * fmsgs1[nK-(K+1)+k];
            for (int k = 0; k < K; k++)  b1 += bmsgs1[nK+k+1] * fmsgs1[nK-(K+1)+k];
            b1 *= exp_node_pots_b[nn];
          } else {
            b0 = bmsgs1[nK];
            b1 = bmsgs1[nK+1] * exp_node_pots_b[nn];
          }
          result_margs[bN+nn] = b1/(b0+b1);
          //cout << result_margs[bN+nn] << " ";
        }
        
        cum_count += Nc;
      }
    }
  }
}


int main(int argc, char **argv) {
  int N = 8;
  int B = 2;
  int C = 3;
  int K = 3;
  int Kp = atoi(argv[1]);

  double *class_counts = new double[C];
  class_counts[0] = 3; class_counts[1] = 2; class_counts[2] = 3;

  double *exp_node_pots = new double[N*B];
  for (int n = 0; n < N; n++) {
    for (int b = 0; b < B; b++) {
      exp_node_pots[b*N+n] = b+n+1; //(nb+1.0)/(N*B);
    }
  }
  double *ys = new double[N];
  for (int i = 0; i < 3; i++) ys[i] = 0;
  for (int i = 3; i < 5; i++) ys[i] = 1;
  for (int i = 5; i < N; i++) ys[i] = 2;

  bool *assn = new bool[N];
  double Z_brute = 0;
  double *margs = new double[N];
  for (int n = 0; n < N; n++) margs[n] = 0;
  
  for (int i = 0; i < pow(2, N); i++) {
    int cur = i;
    int n_on = 0;
    double pot = 1;
    int n_cls0 = 0;
    for (int n = 0; n < N; n++) {
      assn[n] = cur % 2;
      cur = cur / 2;
      n_on += assn[n];

      if (n < class_counts[0] && assn[n])  n_cls0++;
      if (assn[n]) pot *= exp_node_pots[n];
    }

    if (Kp == -1) {
      if (n_on != K)  continue;
    } else if (Kp == -2) {
      if (n_on != K)  continue;
      if (n_cls0 < (K+1)/2)  continue;
    } else {
      if (n_on != K)  continue;
      if (n_cls0 != Kp)  continue;
    }

    //for (int n = 0; n < N; n++) cout << assn[n];
    //cout << " " << pot << endl;
    for (int n = 0; n < N; n++) margs[n] += assn[n] * pot;
    Z_brute += pot;
  }

  cout << endl << "Margs (brute force): ";
  for (int n = 0; n < N; n++) cout << margs[n]/Z_brute << " ";
  cout << endl << "log Z (brute force) = " << log(Z_brute) << endl << endl;

  double *fmsgs1 = new double[(N+1)*(K+1)];
  double *bmsgs1 = new double[(N+1)*(K+1)];
  double *fmsgs2 = new double[(C+1)*(K+1)];
  double *bmsgs2 = new double[(C+1)*(K+1)];

  double *result_margs = new double[B*N];
  double *result_log_Zs = new double[B];
  
  infer(N, B, K, Kp, class_counts, C, exp_node_pots, ys,
        fmsgs1, bmsgs1, fmsgs2, bmsgs2, result_margs, result_log_Zs);


  for (int b = 0; b < B; b++) {
    cout << "BATCH " << b << endl;
    int bN = b*N;
    double sum = 0;
    cout << "LOG Z = " << result_log_Zs[b] << endl;
    cout << "MARGINALS" << endl;
    for (int n = 0; n < N; n++) {
      cout << result_margs[bN + n] << " ";
      sum += result_margs[bN + n];
    }
    cout << endl << endl;
  }
  cout << endl;

  delete[] class_counts;
  delete[] exp_node_pots;
  delete[] ys;
  delete[] assn;
  delete[] margs;
  delete[] fmsgs1;
  delete[] fmsgs2;
  delete[] bmsgs1;
  delete[] bmsgs2;
  delete[] result_margs;
  delete[] result_log_Zs;
}
