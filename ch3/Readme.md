# 说明
该章主要复习wordcount部分，wordcount的代码如下：


```
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]): Unit = {

    var conf = new SparkConf().setAppName("wordcount").setMaster("local")
    var sc = new SparkContext(conf)
    var text = sc.textFile("/root/BigData/spark-2.3.0-bin-hadoop2.7/LICENSE")
    var result = text.flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).sortBy(_._2)
    result.foreach(println)
}
}
```