#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"
#include <string>
#include <fstream>
#include <vector>
#include<iostream>
#include <omp.h>
#include <sstream>
#include <time.h>
#define module_count 5
using namespace std;
int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;





char * string_to_chars(string &x){	
	char *LINE = new char[x.size()+1];
	strcpy(LINE,x.c_str());
	return LINE;
}

vector <string> test;
vector <int> true_result;
void read_testing_set(const char *f){
	ifstream fin(f);
	if(!fin) {cout<<"open file  error\n";exit(1);}	
	string tmp;	
	while(!fin.eof()){
		getline(fin,tmp);
		test.push_back(tmp);
		if(tmp[0]=='+') true_result.push_back(1);
		else true_result.push_back(-1);
	}
	cout<<"testing set of size "<<test.size()<<" have been read into memory\n";
}
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

int max_nr_attr[40000];  
int Max[40000];  
int Modul[module_count][40000];
struct model* model_[module_count]; 
struct feature_node *x[40000];
 

string itoa(int x){
	stringstream s;
	s<<x;
	string y;
	s>>y;
	return y;
}
int atoi(string &x){
	stringstream s;
	s<<x;
	int y;
	s>>y;
	return y;
}

void read_test_into_feature(){

	int total = 0; 

	 
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=5000;

	int linenum=0; 
	while(linenum < test.size()-1)
	{
		max_nr_attr[linenum]=64;
		x[linenum] = (struct feature_node *) malloc(max_nr_attr[linenum]*sizeof(struct feature_node));
		
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr,*helper;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok_r(string_to_chars(test[linenum])," \t\n",&helper);
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr[linenum]-2)	// need one more for index = -1
			{
				max_nr_attr[linenum] *= 2;
				x[linenum] = (struct feature_node *) realloc(x[linenum],max_nr_attr[linenum]*sizeof(struct feature_node));
			}

			idx = strtok_r(NULL,":",&helper);
			val = strtok_r(NULL," \t",&helper);

			if(val == NULL)
				break;
			errno = 0;
			x[linenum][i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[linenum][i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[linenum][i].index;

			errno = 0;
			x[linenum][i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[linenum][i].index <= nr_feature)
				++i;
		} 
		x[linenum][i].index = -1;
		linenum++;
	}
	
}
 
void do_predict(int X)
{
	 
	int total = 0;
	int correct=0; 
	int linenum=0; 
	double predict_label;
	double target_label;
	while(linenum < test.size()-1)
	{
		
		 
		target_label = (double)true_result[linenum];
		predict_label = predict(model_[X],x[linenum]);
		if(predict_label>0)	Modul[X][linenum] = 1;
		else Modul[X][linenum] = -1;

		if(target_label==predict_label)  correct++;
		 
		++total;

		linenum++;
	}
	 
	printf("Module %d Accuracy = %g%% (%d/%d)\n",X+1,(double) correct/total*100,correct,total);
	 
	
}
 

int main(int argc, char **argv)
{
	clock_t start_time;// time 
	start_time = clock(); 
	int i,j,k;
	read_testing_set("traindata/test.txt");
	read_test_into_feature();
	  
	//#pragma omp parallel for num_threads(module_count)   
	for(k=0;k<module_count;k++){
			string M;
		 
			M = "model/M";
			M+=itoa(k+1);
		 
			if((model_[k]=load_model(M.c_str()))==0)
			{
				fprintf(stderr,"can't open model file %s\n",M.c_str());
				exit(1);
			}
			
					 
			do_predict(k);
			free_and_destroy_model(&model_[k]); 
	 				
	} 
	 
	 
	cout<<"Module finished\n";
	
	 
	int total = (int)(test.size()-1);
	int correct=0;
	int vote=0;	

	for(k=0;k<total;k++)	free(x[k]); 

	for(k=0;k<total;k++){
	
		for(i=0;i<module_count;i++){
			vote+=Modul[i][k];
			 
		}
		if(vote>0) Max[k]=1; 
		else Max[k]=-1;
		vote=0;
	}
	for(i=0;i<total;i++) if(Max[i]==true_result[i]) correct++;
	printf("Vote Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
	clock_t end_time = clock();// time 	
	printf("Time consumed: %ld s\n",(end_time-start_time)/CLOCKS_PER_SEC); 
	
	return 0;
}

