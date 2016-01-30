#include"tools.h"
void loadMatrix(string fileName,Mat& m)
{
    vector<double> vals;
    int rows=0;
    ifstream in(fileName);
    string line;
    getline(in, line);
    while (in.good()) {
        rows++;
        stringstream ss;
        ss<<line;
        double v;
        while(ss.good())
        {
            ss>>v;
            // cout<<v<<" ";
            vals.push_back(v);
        }
        // cout<<endl;
        getline(in, line);
    }
    in.close();
    int cols=(int)vals.size()/rows;

    m.create(rows,cols,CV_64F);
    for (int i=0; i<rows; i++) {
        for(int j=0;j<cols;j++)
        {
            m.at<double>(i,j)=vals[i*cols+j];
        }
    }
}

