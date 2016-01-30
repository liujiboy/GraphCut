#include <iostream>
#include "graph.h"
#include<opencv2/opencv.hpp>
#include"tools.h"
#include<vector>
using namespace std;
using namespace cv;
typedef Graph<double, double, double> GraphDouble;
//像素点索引
class idx{
public:
    int row;
    int col;
    idx(int row,int col){
        this->row=row;
        this->col=col;
    }
};
//图割算法
class GraphCut{
private:
    Mat disp; //视差图
    Mat i1; //左视图
    Mat i2; //右视图
    Mat truedisp;   //真实视差图
    vector<vector<vector<double> > >dmaps; //视差空间图像（DSI,存储每个点视差从0到max_disparity的代价值）
    int width; //视图宽度
    int height; //视图高度
    int r=5; //窗口半径
    int max_disparity=28; //最大视差值
    int border_size=18; //图像的边框大小（此处不计算视差）
    //ncc cost function
    //返回(row,col)处像素点，视差为d时的cost value
    double ncc(int row,int col,int d)
    {
        int col1 = col;
        int row1 = row;
        int col2 = col1 - d;
        int row2 = row1;
        if (col1 - r >= 0 && col1 + r < width && col2 - r >= 0 && col2 + r < width
            && row1 - r >= 0 && row1 + r < height && row2 - r >= 0 && row2 + r < height)
        {
            Mat cl;
            i1(Rect(col1-r,row1-r,2*r+1,2*r+1)).copyTo(cl);
            cl=cl.reshape(1,1);
            Mat cr;
            i2(Rect(col2-r,row2-r,2*r+1,2*r+1)).copyTo(cr);
            cr=cr.reshape(1,1);
            cl=cl-mean(cl);
            cr=cr-mean(cr);
            return 1-cl.dot(cr)/(norm(cl)*norm(cr));
        }else
        {
            return 2;
        }
    }
    //视差值fa和视差值fb的罚函数
    double penalty(double fp,double fq)
    {
        double e=abs(fp-fq);
        return e>1?1:e;
    }
    //能量函数的平滑项
    double smooth(int row,int col)
    {
        vector<idx> neighbor;
        get_neighbor(col,row,neighbor);
        double fp=disp.at<double>(row,col);
        double sum=0;
        for(vector<idx>::iterator q=neighbor.begin();q!=neighbor.end();q++)
        {
            
            double fq=disp.at<double>(q->row,q->col);
            sum+=penalty(fp, fq);
        }
        return sum;
    }
    //总能量函数，即待优化的函数
    double energy()
    {
        double e=0;
        //由于测试好数据存在宽度18的边框，故此略过这个边框
        for(int row=border_size;row<height-border_size;row++)
            for(int col=border_size;col<width-border_size;col++)
            {
                e+=dmaps[row][col][disp.at<double>(row,col)]+smooth(row,col);
            }
        return e;
    }
    //创建DSI
    void buildDmaps()
    {
        dmaps.resize(height,vector<vector<double> >(width,vector<double>(max_disparity+1,0)));
        for(int row=0;row<height;row++)
            for(int col=0;col<width;col++)
                for(int d=0;d<=max_disparity;d++)
                {
                    dmaps[row][col][d]=ncc(row,col,d);
                }
    }
    //返回(row,col)的邻居索引,默认返回周围的8邻居
    void get_neighbor(int row,int col,vector<idx>&neighbor,int s=1)
    {
        for (int i = -s; i <= s; i++)
            for (int j = -s; j <= s; j++)
            {
                int col1=col+j;
                int row1=row+i;
                if (row1!=row&&col1!=col) {
                    if(col1>=0&&col1<width&&row1>=0&&row1<height)
                    {
                        
                        neighbor.push_back(idx(row1, col1));
                    }
                }
            }
    }
    //a和b是否是邻居
    bool is_neighbor(idx a,idx b,int s=1)
    {
        for (int i = -s; i <= s; i++)
            for (int j = -s; j <= s; j++)
            {
                
                if(i!=0&&j!=0)
                {
                    if(b.row==a.row+i &&b.col==a.col+j)
                        return true;
                }
            }
        return false;
    }
    //查找视差值等于v（flag=true）或者不等于v（flag=false）的点位置
    void find_idx(int v,vector<idx>& idxvec,bool flag=true)
    {
        
        for (int row=border_size; row<height-border_size; row++) {
            for (int col=border_size; col<width-border_size; col++) {
                if (flag) {
                    if (disp.at<double>(row,col)==v) {
                        idxvec.push_back(idx(row,col));
                    }
                }
                else
                {
                    if (disp.at<double>(row,col)!=v) {
                        idxvec.push_back(idx(row,col));
                    }
                }
            }
        }
    }
    //打印错误视差百分比
    void print_bad_points_percentage()
    {
        double n=width*height;
        Rect roi(border_size,border_size,width-2*border_size,height-2*border_size);
        double bad_points=countNonZero(abs(truedisp(roi)-disp(roi))>1);
        cout<<"错误视差百分比:"<<bad_points/n*100<<endl;
    }
    //alpha beta swap
    void alpha_beta_swap(int max_iter=10)
    {
        Mat backup;
        disp.copyTo(backup);
        bool success=true;
        int iter=0;
        double old_energy=energy();
        cout<<"初始能量为"<<old_energy<<endl;
        while (success&&iter<max_iter) {
            success=false;
            for (int alpha=0; alpha<max_disparity; alpha++) {
                for(int beta=alpha+1;beta<=max_disparity;beta++)
                {
                    
                    cout<<"交换："<<alpha<<","<<beta<<endl;
                    vector<idx> alpha_idxvec,beta_idxvec;
                    vector<int> alpha_node_idxvec,beta_node_idxvec;
                    find_idx(alpha,alpha_idxvec);
                    find_idx(beta,beta_idxvec);
                    GraphDouble g(width*height,width*height);
                    //设置顶点和权重
                    //设置alpha顶点
                    for (vector<idx>::iterator it=alpha_idxvec.begin();it!=alpha_idxvec.end();it++) {
                        int node_id=g.add_node();
                        double source_cap=dmaps[it->row][it->col][alpha];
                        double sink_cap=dmaps[it->row][it->col][beta];
                        g.add_tweights(node_id,source_cap,sink_cap);
                        alpha_node_idxvec.push_back(node_id);
                    }
                    //设置beta顶点
                    for (vector<idx>::iterator it=beta_idxvec.begin();it!=beta_idxvec.end();it++) {
                        int node_id=g.add_node();
                        double source_cap=dmaps[it->row][it->col][alpha];
                        double sink_cap=dmaps[it->row][it->col][beta];
                        g.add_tweights(node_id,source_cap,sink_cap);
                        beta_node_idxvec.push_back(node_id);
                    }
                    //设置相邻点
                    for (int i=0; i<alpha_idxvec.size();i++) {
                        idx alpha_id=alpha_idxvec[i];
                        for (int j=0; j<beta_idxvec.size(); j++) {
                            idx beta_id=beta_idxvec[j];
                            if(is_neighbor(alpha_id,beta_id))
                            {
                                double cap=penalty(alpha,beta);
                                g.add_edge(alpha_node_idxvec[i], beta_node_idxvec[j], cap, cap);
                            }
                        }
                    }
                    //计算最大流
                    g.maxflow();
                    //交换alpha和beta
                    for (int i=0; i<alpha_idxvec.size(); i++) {
                        idx alpha_id=alpha_idxvec[i];
                        int node_id=alpha_node_idxvec[i];
                        if(g.what_segment(node_id)== GraphDouble::SOURCE)
                        {
                            disp.at<double>(alpha_id.row,alpha_id.col)=beta;
                        }
                    }
                    for (int i=0; i<beta_idxvec.size(); i++) {
                        idx beta_id=beta_idxvec[i];
                        int node_id=beta_node_idxvec[i];
                        if(g.what_segment(node_id)== GraphDouble::SINK)
                        {
                            disp.at<double>(beta_id.row,beta_id.col)=alpha;
                        }
                    }
                    double new_energy=energy();
                    cout<<"优化后能量为："<<new_energy<<endl;
                    if(old_energy>new_energy)
                    {
                        cout<<"优化成功，交换"<<alpha<<","<<beta<<endl;
                        old_energy=new_energy;
                        print_bad_points_percentage();
                        disp.copyTo(backup);
                        success=true;
                    }else
                    {
                        cout<<"优化失败"<<endl;
                        backup.copyTo(disp);
                    }
                    
                }
                
            }
            cout<<"第"<<iter<<"次迭代，优化后能量为"<<old_energy<<endl;
            print_bad_points_percentage();
            iter++;
        }
    }
    //α expansion
    void alpha_expansion(int max_iter=10)
    {
        Mat backup;
        disp.copyTo(backup);
        bool success=true;
        int iter=0;
        double INF=1000000;
        double old_energy=energy();
        cout<<"初始能量为"<<old_energy<<endl;
        while (success&&iter<max_iter) {
            success=false;
            for (int alpha=0; alpha<=max_disparity; alpha++) {
                vector<idx> alpha_idxvec,non_alpha_idxvec;
                vector<int> alpha_node_idxvec,non_alpha_node_idxvec;
                find_idx(alpha, alpha_idxvec);
                find_idx(alpha, non_alpha_idxvec,false);
                GraphDouble g(width*height,width*height);
                //设置alpha顶点
                for (vector<idx>::iterator it=alpha_idxvec.begin();it!=alpha_idxvec.end();it++) {
                    int node_id=g.add_node();
                    double source_cap=dmaps[it->row][it->col][alpha];
                    double sink_cap=INF;
                    g.add_tweights(node_id,source_cap,sink_cap);
                    alpha_node_idxvec.push_back(node_id);
                }
                //设置非alpha顶点
                for (vector<idx>::iterator it=non_alpha_idxvec.begin();it!=non_alpha_idxvec.end();it++) {
                    int node_id=g.add_node();
                    double source_cap=dmaps[it->row][it->col][alpha];
                    int d=disp.at<double>(it->row,it->col);
                    double sink_cap=dmaps[it->row][it->col][d];
                    g.add_tweights(node_id,source_cap,sink_cap);
                    non_alpha_node_idxvec.push_back(node_id);
                }
                //设置相邻点
                for (int i=0; i<alpha_idxvec.size();i++) {
                    idx alpha_id=alpha_idxvec[i];
                    for (int j=0; j<non_alpha_idxvec.size(); j++) {
                        idx non_alpha_id=non_alpha_idxvec[j];
                        if(is_neighbor(alpha_id,non_alpha_id))
                        {
                            double fp=disp.at<double>(alpha_id.row,alpha_id.col);
                            double fq=disp.at<double>(non_alpha_id.row,non_alpha_id.col);
                            if (fp==fq) {
                                double cap=penalty(fp,fq);
                                g.add_edge(alpha_node_idxvec[i], non_alpha_node_idxvec[j], cap, cap);
                            }else
                            {
                                int node_id=g.add_node();
                                double cap1=penalty(fp,alpha);
                                double cap2=penalty(alpha,fq);
                                g.add_edge(alpha_node_idxvec[i], node_id, cap1, cap1);
                                g.add_edge(node_id, non_alpha_node_idxvec[j], cap2, cap2);
                                double source_cap=0;
                                double sink_cap=penalty(fp,fq);
                                g.add_tweights(node_id, source_cap, sink_cap);
                            }
                            
                        }
                    }
                }
                g.maxflow();
                //  cout<<g.maxflow()<<endl;
                cout<<"扩张"<<alpha<<endl;
                for (int i=0; i<non_alpha_idxvec.size(); i++) {
                    idx non_alpha_id=non_alpha_idxvec[i];
                    int node_id=non_alpha_node_idxvec[i];
                    if(g.what_segment(node_id)== GraphDouble::SINK)
                    {
                        disp.at<double>(non_alpha_id.row,non_alpha_id.col)=alpha;
                    }
                }
                double new_energy=energy();
                cout<<"优化后能量为："<<new_energy<<endl;
                if(old_energy>new_energy)
                {
                    cout<<"优化成功"<<endl;
                    print_bad_points_percentage();
                    old_energy=new_energy;
                    disp.copyTo(backup);
                    success=true;
                }else
                {
                    cout<<"优化失败恢复"<<endl;
                    backup.copyTo(disp);
                }
            }
            cout<<"第"<<iter<<"次迭代，优化后能量为"<<old_energy<<endl;
            print_bad_points_percentage();
            iter++;
        }
    }
public:
    GraphCut(const char* disp_filename,const char* truedisp_filename,const char* i1_filename,const char* i2_filename)
    {
        cout<<"加载数据"<<endl;
        loadMatrix(disp_filename,disp);
        loadMatrix(truedisp_filename,truedisp);
        width=disp.cols;
        height=disp.rows;
        loadMatrix(i1_filename,i1);
        loadMatrix(i2_filename,i2);
        print_bad_points_percentage();
        cout<<"创建DSI"<<endl;
        buildDmaps();
        alpha_beta_swap();
        alpha_expansion();
        
    }
    
};

int main()
{
    GraphCut g("data/disp.txt","data/truedisp.txt","data/i1.txt","data/i2.txt");
    
   
  
  
    
//cout<<ncc(0,0,5)<<endl;
//cout<<ncc(100,100,5)<<endl;
//    typedef Graph<int,int,int> GraphType;
//    GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
//
//    g -> add_node();
//    g -> add_node();
//
//    g -> add_tweights( 0,   /* capacities */  1, 5 );
//    g -> add_tweights( 1,   /* capacities */  2, 6 );
//    g -> add_edge( 0, 1,    /* capacities */  3, 4 );
//
//    int flow = g -> maxflow();
//    printf("Flow = %d\n", flow);
//    printf("Minimum cut:\n");
//    if (g->what_segment(0) == GraphType::SOURCE)
//        printf("node0 is in the SOURCE set\n");
//    else
//        printf("node0 is in the SINK set\n");
//    if (g->what_segment(1) == GraphType::SOURCE)
//        printf("node1 is in the SOURCE set\n");
//    else
//        printf("node1 is in the SINK set\n");
//
//    delete g;

return 0;
}
