import java.util.ArrayList;
import java.util.List;

/**
 * HughesMatrix.java
 *
 * This file is a rather abstract matrix implementation to make math with matricies much easier throughout the neural
 * network later on. The necesary functions include: addition, subtraction, transposition, and multiplication. All our
 * matricies will use doubles.
 *
 * Authors: Jack Hughes
 * Date: 2-24-21
 * -JBH
 */

class HughesMatrix {
    //INSTANCE-DATA
    double [][]data;
    int rows,cols;

    //CONSTRUCTOR (initialize a HughesMatrix of a given size)
    public HughesMatrix(int rows,int cols) {
        data= new double[rows][cols];
        this.rows=rows;
        this.cols=cols;
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                data[i][j]=Math.random()*2-1;
            }
        }
    }

    //METHODS
    //print(): print this HughesMatrix to the console
    public void print()
    {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                System.out.print(this.data[i][j]+" ");
            }
            System.out.println();
        }
    }
    //add(int scalar): add the given scalar to every entry in this hughesMatrix
    public void add(int scalar)
    {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                this.data[i][j]+=scalar;
            }

        }
    }
    //add(HughesMatrix m): add the given HughesMatrix to this one.
    public void add(HughesMatrix m)
    {
        if(cols!=m.cols || rows!=m.rows) {
            System.out.println("Shape Mismatch");
            return;
        }

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                this.data[i][j]+=m.data[i][j];
            }
        }
    }
    //fromArray(double[]x): create and return a HughesMatrix from a double array
    public static HughesMatrix fromArray(double[]x)
    {
        HughesMatrix temp = new HughesMatrix(x.length,1);
        for(int i =0;i<x.length;i++)
            temp.data[i][0]=x[i];
        return temp;

    }
    //toList(): create and return a list made from this HughesMatrix
    public List<Double> toList() {
        List<Double> temp= new ArrayList<Double>()  ;

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                temp.add(data[i][j]);
            }
        }
        return temp;
    }
    //subtract(HM a, HM b): subtract two HughesMatrix objects-- "a-b" -- returning the resultant HughesMatrix
    public static HughesMatrix subtract(HughesMatrix a, HughesMatrix b) {
        HughesMatrix temp=new HughesMatrix(a.rows,a.cols);
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                temp.data[i][j]=a.data[i][j]-b.data[i][j];
            }
        }
        return temp;
    }
    //clone(HM a): return a clone of the given HughesMatrix
    public static HughesMatrix clone(HughesMatrix a) {
        HughesMatrix temp=new HughesMatrix(a.cols,a.rows);
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                temp.data[j][i]=a.data[i][j];
            }
        }
        return temp;
    }
    //multiply(HM a, HM b): return a HM resulting from the multiplication of "a * b"
    public static HughesMatrix multiply(HughesMatrix a, HughesMatrix b) {
        HughesMatrix temp=new HughesMatrix(a.rows,b.cols);
        for(int i=0;i<temp.rows;i++)
        {
            for(int j=0;j<temp.cols;j++)
            {
                double sum=0;
                for(int k=0;k<a.cols;k++)
                {
                    sum+=a.data[i][k]*b.data[k][j];
                }
                temp.data[i][j]=sum;
            }
        }
        return temp;
    }
    //multiply(HM a): multiply this HM by a
    public void multiply(HughesMatrix a) {
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                this.data[i][j]*=a.data[i][j];
            }
        }

    }
    //multiply(double a): multiply this HM's values by a
    public void multiply(double a) {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                this.data[i][j]*=a;
            }
        }

    }
    //initSigmoid(): apply the sigmoid function to this HM
    public void initSigmoid() {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                this.data[i][j] = 1/(1+Math.exp(-this.data[i][j]));
        }

    }
    //deriveSigmoid(): remove the sigmoid function from this HM
    public HughesMatrix deriveSigmoid() {
        HughesMatrix temp=new HughesMatrix(rows,cols);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                temp.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
        }
        return temp;

    }
}