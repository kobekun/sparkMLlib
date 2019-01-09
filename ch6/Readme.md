# 说明
本章主要讲解基础统计部分，包括基本统计、假设检验、相关系数等

## 数据集
数据集有两个文件，分别是：

1. beijing.txt 北京历年降水量，不带年份
2. beijing2.txt 北京历年降水量，带年份

## 源代码
源代码比较少，故在此给出：

### 基础统计

```
val txt = sc.textFile("beijing.txt")
val data = txt.flatMap(_.split(",")).map(value => Vectors.dense(value.toDouble))
Statistics.colStats(data)
```

### 一致性

```
val txt = sc.textFile("beijing2.txt")
val data = txt.flatMap(_.split(",")).map(_.toDouble)
val years = data.filter(_>1000)
val values = data.filter(_<=1000)
Statistics.corr(years,values)

```

### 假设检验


	        男，女
	右利手 127,147
	左利手 19,10

```
Statistics.chiTest(Matrices.dense(2,2,Array(127,19,147,10)))

```