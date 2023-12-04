#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x*(sizeX*sizeY*sizeZ) + y*(sizeY*sizeZ) + z*sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x*(sizeX*sizeY*sizeZ) + y*(sizeY*sizeZ) + z*sizeZ + b] = val; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        //loop over Heads
        for (int h = 0; h < H; h++) {
            //loop over Sequence Length
            for (int qkt1 = 0; qkt1 < N; qkt1++){
                for (int qkt2 = 0; qkt2 < N; qkt2++){
                    float val_qkt = 0.0;
                    for(int j = 0; j < d; j++){
                        float val_q = fourDimRead(Q, b, h, qkt1, j, H, N, d);
                        float val_kt = fourDimRead(K, b, h, qkt2, j, H, N, d);
                        val_qkt += val_q * val_kt;                        
                    }
                    //printf("val_qkt = %f\n", val_qkt); 
                    twoDimWrite(QK_t, qkt1, qkt2, N, val_qkt);
                }
            }
            for (int qkt1 = 0; qkt1 < N; qkt1++){
                float sum_exp_val_qkt = 0.0;
                for (int qkt2 = 0; qkt2 < N; qkt2++){
                    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
                    float exp_val_qkt = exp(val_qkt);
                    sum_exp_val_qkt += exp_val_qkt;
                }
                for (int qkt2 = 0; qkt2 < N; qkt2++){
                    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
                    float exp_val_qkt = exp(val_qkt);
                    float val_softmax = exp_val_qkt / sum_exp_val_qkt;
                    //printf("exp_val_qkt = %f ", exp_val_qkt);      
                    //printf("sum_exp_val_qkt = %f ", sum_exp_val_qkt);      
                    //printf("val_softmax = %f ", val_softmax);    
                    twoDimWrite(QK_t, qkt1, qkt2, N, val_softmax);
                }
            }

            for(int o1 = 0; o1 < N; o1++){
                for(int o2 = 0; o2 < d; o2++){
                    float val_o = 0.0;
                    for (int j = 0; j < N; j++){
                        float val_softmax = twoDimRead(QK_t, o1, j, N);
                        float val_v = fourDimRead(V, b, h, j, o2, H, N, d);
                        val_o += val_softmax * val_v;
                    }
                    
                    fourDimWrite(O, b, h, o1, o2, H, N, d, val_o);
                }
            }           
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    int L = 16;

    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            //for (int i = 0; i < N ; i++){
	    std::fill(QK_t.begin(), QK_t.end(), 0);

	    for (int qkt1_b = 0; qkt1_b < N; qkt1_b = qkt1_b + L){
	    	for (int qkt2_b = 0; qkt2_b < N; qkt2_b = qkt2_b + L){  
		    for( int j_b = 0; j_b < d; j_b = j_b + L){  
		    	for (int qkt1 = qkt1_b; qkt1 < std::min(qkt1_b + L, N); qkt1++){
			    for (int qkt2 = qkt2_b; qkt2 < std::min(qkt2_b + L, N); qkt2++){
			    	float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
			        for(int j = j_b; j < std::min(j_b + L,  d); j++){
				    float val_q = fourDimRead(Q, b, h, qkt1, j, H, N, d);
				    float val_kt = fourDimRead(K, b, h, qkt2, j, H, N, d);
				    val_qkt += val_q * val_kt;                        
			     	}
			    //printf("val_qkt = %f", val_qkt);
			        twoDimWrite(QK_t, qkt1, qkt2, N, val_qkt);
			    }
		    	}
		    }
	    	}
	    }

	    
	
	    for (int qkt1 = 0; qkt1 < N; qkt1++){
	        float sum_exp_val_qkt = 0.0;
	        for (int qkt2 = 0; qkt2 < N; qkt2++){
	 	    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
		    float exp_val_qkt = exp(val_qkt);
		    sum_exp_val_qkt += exp_val_qkt;
	        }
	        for (int qkt2 = 0; qkt2 < N; qkt2++){
		    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
		    float exp_val_qkt = exp(val_qkt);
		    float val_softmax = exp_val_qkt / sum_exp_val_qkt;
		    //printf("exp_val_qkt = %f ", exp_val_qkt);      
		    //printf("sum_exp_val_qkt = %f ", sum_exp_val_qkt);      
		    //printf("val_softmax = %f ", val_softmax);    
		    twoDimWrite(QK_t, qkt1, qkt2, N, val_softmax);
	        }
	    }
	
	    for (int o1_b = 0; o1_b < N; o1_b = o1_b + L){
                for (int o2_b = 0; o2_b < d; o2_b = o2_b + L){  
		    for( int j_b = 0; j_b < N; j_b = j_b + L){  
		        for (int o1 = o1_b; o1 < std::min(o1_b + L, N); o1++){
			    for (int o2 = o2_b; o2 < std::min(o2_b + L, d); o2++){
			        float val_o = fourDimRead(O, b, h, o1, o2, H, N, d);
			        for(int j = j_b; j < std::min(j_b + L,  N); j++){
				    float val_softmax = twoDimRead(QK_t, o1, j, N);
				    float val_v = fourDimRead(V, b, h, j, o2, H, N, d);
				    val_o += val_softmax * val_v;                        
			        }
			//printf("val_qkt = %f", val_qkt);
			        fourDimWrite(O, b, h, o1, o2, H, N, d, val_o);
			    }
		        }
		    }
	        }
 	    }           
	
	}

    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    //

    #pragma omp parallel for collapse(3) 
    for(int b = 0; b < B; b++){
	for(int h = 0; h < H; h++){
	    for(int i = 0; i < N; i++){
		//printf("Starting New Row of Computation\n");	
    		at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
		std::vector<float> ORow = formatTensor(ORowTensor);
		int zeroRow = 0;
	    	//std::fill(ORow.begin(), ORow.end(), 0);
		for (int q = 0; q < N; q++){
                    float val_qkt = 0.0;
                    for(int j = 0; j < d; j++){
                        float val_q = fourDimRead(Q, b, h, i, j, H, N, d);
			//printf("Read q");
                        float val_kt = fourDimRead(K, b, h, q, j, H, N, d);
			//printf("Read KT");
                        val_qkt += val_q * val_kt;                        
                    }
                    //printf("val_qkt = %f\n", val_qkt); 
                    twoDimWrite(ORow, zeroRow, q, N, val_qkt);
                }
		//A single row has been completed (matmul)
	 	//printf("A Row has completed its computation\n");	

		float sum_exp_val_qkt = 0.0;
                for (int q = 0; q < N; q++){
                    float val_qkt = twoDimRead(ORow, zeroRow, q, N);
                    float exp_val_qkt = exp(val_qkt);
                    sum_exp_val_qkt += exp_val_qkt;
                }
                for (int q = 0; q < N; q++){
                    float val_qkt = twoDimRead(ORow, zeroRow, q, N);
                    float exp_val_qkt = exp(val_qkt);
                    float val_softmax = exp_val_qkt / sum_exp_val_qkt;
                    twoDimWrite(ORow, zeroRow, q, N, val_softmax);
                }
		//Softmax has been completed	
		//printf("Softmax of a row has completed\n");

                for(int o = 0; o < d; o++){
                    float val_o = 0.0;
                    for (int j = 0; j < N; j++){
                        float val_softmax = twoDimRead(ORow, zeroRow, j, N);
                        float val_v = fourDimRead(V, b, h, j, o, H, N, d);
                        val_o += val_softmax * val_v;
                    }
                    
                    fourDimWrite(O, b, h, i, o, H, N, d, val_o);
                }
		//printf("Finished writing to output\n");
	    }
	   
	}
    }    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    int zeroRow = 0;

    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
            //for (int i = 0; i < N ; i++){
	    
            std::fill(l.begin(), l.end(), 0);
            for (int col_block = 0; col_block < N; col_block = col_block + Bc){  
            /*for(int col = col_block; col < std::min(col_block+BC,N); col++){
                for(int j = 0; j < d; j++){
                    float val_k = fourDimRead(Q, b, h, col, j, H, N, d); 
                float val_v = fourDimRead(V, b, h, j, col, H, N, d);
                twoDimWrite(Kj, col, j, val_k);
                twoDimWrite(Vj, col, j, val_v);
                }
            }*/
                
                //printf("Beginning new col_block %i\n", col_block);
                for (int row_block = 0; row_block < N; row_block = row_block + Br){
                    std::fill(Sij.begin(), Sij.end(), 0);
                    //printf("Beginning new row_block %i\n", row_block);
                    std::fill(lij.begin(), lij.end(), 0);
                    //load li, Oi
                    std::fill(li.begin(), li.end(), 0);
                    std::fill(Oi.begin(), Oi.end(), 0);
                    for (int row = row_block; row < std::min(row_block + Br, N); row++){
                        int true_row = row - row_block;
                        li[true_row] = l[row];
                        for(int j = 0; j < d; j++){
                            float val_oi = fourDimRead(O, b, h, row, j, H, N, d);
                            twoDimWrite(Oi, true_row, j, d, val_oi);
                        }
                    }
                    for (int row = row_block; row < std::min(row_block + Br, N); row++){
                        int true_row = row - row_block;
                        //printf("Beginning new row %i\n", row);
                        float val_li = li[true_row];
                        float val_lij = lij[true_row];
                        for (int col = col_block; col < std::min(col_block + Bc, N); col++){
                            int true_col = col - col_block;
                            //printf("Beginning new col %i\n", col);
                            float val_qkt = twoDimRead(Sij, true_row, true_col, Bc);
                            //printf("Read in Sij for row %i, col %i\n", row, col);
                            for(int j = 0; j < d; j++){
                                float val_q = fourDimRead(Q, b, h, row, j, H, N, d);
                                float val_kt = fourDimRead(K, b, h, col, j, H, N, d);
                                val_qkt += val_q * val_kt;
                            }
                            //printf("Calculated qkt\n");
                            twoDimWrite(Sij, true_row, true_col, Bc, val_qkt);
                            float exp_val_qkt = exp(val_qkt);
                            twoDimWrite(Pij, true_row, true_col, Bc, exp_val_qkt);
                            val_lij += exp_val_qkt;
                            lij[true_row] = val_lij;

                        }
                    //printf("Completed rowsum of exp\n");
                        float my_lnew = val_li + lij[true_row];
                        lnew[true_row] = my_lnew;
                        for(int j = 0; j < d; j++){
                            float O_i_to_read = twoDimRead(Oi, true_row, j, d); //This could be wrong here!!!					   
                            float output_sum = 0.f;
                            for (int col = col_block; col < std::min(col_block + Bc, N); col++){
                                int true_col = col - col_block;
                                float exp_val = twoDimRead(Pij, true_row, true_col, Bc);
                                float v_val = fourDimRead(V, b, h, col, j, H, N, d);
                                //float v_val = twoDimRead(Vj, col, j, d);
                                output_sum += exp_val * v_val;
                            }
                            float O_to_write = (val_li*O_i_to_read + output_sum)/lnew[true_row];
                            //printf("Writing to output row: %i, j: %i \n", row, j);
                            twoDimWrite(Oi, true_row, j, d, O_to_write);
                            fourDimWrite(O, b, h, row, j, H, N, d, O_to_write);
                        }
                        l[row] = lnew[true_row];
                    }
                }
            }	
            //printf("Bc: %i, Br: %i, B: %i, H: %i, N: %i, d: %i\n", Bc, Br, B, H, N, d);
            //printf("SIZES: O: %i, Q: %i, K: %i, V: %i, Sij: %i, Pij: %i, Kj: %i, Vj: %i, Qi: %i Oi: %i, l: %i, pv: %i, li: %i, lij: %i, lnew: %i\n", O.size(), Q.size(), K.size(), V.size(), Sij.size(), Pij.size(), Kj.size(), Vj.size(), Qi.size(), Oi.size(), l.size(), PV.size(), li.size(), lij.size(), lnew.size());
	    }
    }
	    /*
	    for (int col_block = 0; col_block < N; col_block = col_block + Bc){  
		for (int col = col_block; col < std::min(col_block + Bc, N); col++){
		    for(int j = 0; j < d; j++){

		   	float val_k = fourDimRead(Q, b, h, col, j, H, N, d); 
			float val_v = fourDimRead(V, b, h, j, col, H, N, d);
			twoDimWrite(Kj, col, j, val_k);
			twoDimWrite(Vj, col, j, val_v);
			
		        }

		    for (int row_block = 0; row_block < N; row_block = row_block + Br){
		        for (int row = row_block; row < std::min(row_block + Br, N); row++){
			    //twoDimWrite(li, zeroRow, row, 0.f);
		            for(int j= 0; j < d; j++){
			        float val_q = fourDimRead(Q, b, h, row, j, H, N, d);
			        twoDimWrite(Qi, row, j, val_q);
			        twoDimWrite(Oi, row, j, 0.f);
		  	    }
		        }
	   	    }
    		    //Compute Si_j	
		    for (int row_block = 0; row_block < N; row_block = row_block + Br){
		        for (int row = row_block; row < std::min(row_block + Br, N); row++){
                    	    float val_qkt = 0.0;
		            for(int j= 0; j < d; j++){
			    // SHOULD THIS BE ACCESSED USING N?
			        float val_q = twoDimRead(Qi, row, j, N);
			        float val_k = twoDimRead(Kj, col, j, N)
			        val_qkt += val_1 * val_kt;
			    }
			    twoDimWrite(Sij, row, col, N, val_qkt);
			    float P_ij = exp(val_qkt);
			    twoDimWrite(Pij, row, col, N, val_qkt);
			    float row_sum = twoDimRead(lij, zeroRow, row, N); //row or col?
			    row_sum += P_ij;
			    twoDimWrite(lij, zeroRow, row, N, row_sum);  //row or col?
				
		        }
	   	    }
		}
		
	    }

	    
	
	    for (int qkt1 = 0; qkt1 < N; qkt1++){
	        float sum_exp_val_qkt = 0.0;
	        for (int qkt2 = 0; qkt2 < N; qkt2++){
	 	    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
		    float exp_val_qkt = exp(val_qkt);
		    sum_exp_val_qkt += exp_val_qkt;
	        }
	        for (int qkt2 = 0; qkt2 < N; qkt2++){
		    float val_qkt = twoDimRead(QK_t, qkt1, qkt2, N);
		    float exp_val_qkt = exp(val_qkt);
		    float val_softmax = exp_val_qkt / sum_exp_val_qkt;
		    //printf("exp_val_qkt = %f ", exp_val_qkt);      
		    //printf("sum_exp_val_qkt = %f ", sum_exp_val_qkt);      
		    //printf("val_softmax = %f ", val_softmax);    
		    twoDimWrite(QK_t, qkt1, qkt2, N, val_softmax);
	        }
	    }
	
	    for (int o1_b = 0; o1_b < N; o1_b = o1_b + L){
                for (int o2_b = 0; o2_b < d; o2_b = o2_b + L){  
		    for( int j_b = 0; j_b < N; j_b = j_b + L){  
		        for (int o1 = o1_b; o1 < std::min(o1_b + L, N); o1++){
			    for (int o2 = o2_b; o2 < std::min(o2_b + L, d); o2++){
			        float val_o = fourDimRead(O, b, h, o1, o2, H, N, d);
			        for(int j = j_b; j < std::min(j_b + L,  N); j++){
				    float val_softmax = twoDimRead(QK_t, o1, j, N);
				    float val_v = fourDimRead(V, b, h, j, o2, H, N, d);
				    val_o += val_softmax * val_v;                        
			        }
			//printf("val_qkt = %f", val_qkt);
			        fourDimWrite(O, b, h, o1, o2, H, N, d, val_o);	
			    }
		        }
		    }
	        }
 	    }           
*/	




    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
