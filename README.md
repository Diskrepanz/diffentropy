# diffentropy

# 操作步骤
1. 克隆仓库到本地
```
git clone https://github.com/cyruscyliu/diffentropy.git
```
2. 连接远程仓库
```
git remote add origin git@github.com:cyruscyliu/diffentropy.git
```
3. 提交修改
```
git add *
git commit -m 'description'
git push -u origin master
```
4. 拉取更新
```
git pull origin master:master
```
5. 分支操作
+ 创建并切换到分支
```
git checkout -b yourbranchname

# 等价于以下两条命令
git branch yourbranchname
git checkout yourbranchname
```
+ 查看分支
```
git branch
```
+ 切换回master
```
git checkout master
```
+ 合并新分支到master(必须先切换到master分支)
```
# ff
git merge yourbranchname
# no-ff
git merge --no-ff -m 'merge with no-ff' yourbranchname
```
+ 删除分支
```
git branch -d yourbranchname
```
6. 解决冲突
+ 合并冲突后查看冲突文件
```
git status
```
+ 直接打开文件进行选择
+ 重新添加/提交
+ 合并即完成
+ 删除分支(可选)





