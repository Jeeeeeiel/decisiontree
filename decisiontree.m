function tree = decisiontree(data,colnames,classindex,method)
%DECISIONTREE generate decision tree from data(M*N). 
%NOTE:every Ordinal property should be quantize to keep ordinal information
%in calculation.For example:small,medium,big->1,2,3(2,1,3 is not allowed).
%Variable Defination:
%   data        sample data,size M*N cell, each row in cell stand for one observation.
%   colnames     1*M cell,stores Attribute names responding to every column in data.
%   classindex  specify which column in data representative Category info.
%   tree        decision tree generated based on data.

%colnames = {colnames;zeros(length(colnames))};%mark as unused


% data=textscan(fileid,'%s %f %f %f %f %f %f %f %d','Delimiter',',');


tree = treegrowth(data,colnames,classindex,method);

end

function node=treegrowth(data,colsleft,classindex,method) 
    if teststop(data,colsleft,classindex)==true
        node = struct();
        node.type = 'leaf';
        node.label = classify(data);
    else
        node = struct();
        node.type = 'node';
        [node.attributeindex,node.cond] = findbestsplitattribute(data,colsleft,classindex,method);
        node.attributename = colsleft{attributeindex};
        
        %remove colname used!!
        colsleft(:,attributeindex)=[];
        %remove rowdata for each child
        [datal,datar] = splitdata(data,colsleft,classindex);
        
        node.leftchild = treegrowth(datal,colsleft,classindex);
        node.rightchild = treegrowth(datar,colsleft,classindex);
    end
end

function [datal,datar] = splitdata(data,attributeindex,cond)
    if isnumeric(cond)==true
        mask = cell2mat(data(:,attributeindex))<=cond; %row numbers
        datal = data(mask,[1:attributeindex-1,attributeindex+1:end]);
        datar = data(~mask,[1:attributeindex-1,attributeindex+1:end]);
    else
        mask = mystrcmp(data(:,attributeindex),cond);
        datal = data(mask,[1:attributeindex-1,attributeindex+1:end]);
        datar = data(~mask,[1:attributeindex-1,attributeindex+1:end]);
    end
end

function contains = mystrcmp(strarray,cond) 
%test every str in strarray wether it is in cond
    contains = zeros(length(strarray),1);
    for ii = 1:length(strarray)
        contains(ii) = sum(strcmp(strarray{ii},cond));
    end
    contains = contains>0;
end

function label = classify(data,classindex)
    count = getcountincol(data(:,classindex));
%     if iscell(count)==true
%         [~,i] = max(cell2mat(count(:,end)))
%         label = count{i,1};
%     else
%         [~,i] = max(count(:,end));
%         label = count(i,1);
%     end
    [~,i] = max(count(:,end));
	label = count(i,1);
end

function stop = teststop(data,colsleft,classindex)
    stop = false;
    if size(data,1)<=1
        stop = true;
    elseif length(colsleft)==1  %no col left except class column
        stop = true;
    elseif size(getcountincol(data(:,classindex)),1)==1 %belongs to same class
        stop = true;
    elseif iseveryrowequal((data(:,[1:end-1,end+1:end])))   %same value,exclude category column
        %compare every column except classindex for every row 
        stop = true;
    end
end

function count = getcountincol(rows)    %pass single col
    if isnumeric(rows{1})
       count = tabulate(cell2mat(rows));
    else
       count = tabulate(rows); 
    end
end

function equal = iseveryrowequal(data)%exclude category column
    equal = 1;
    for ii = 1:size(data,1)
        equal = equal*prod(cellfun(@isequal,data(1,:),data(ii,:)));
        if equal == 0
           return; 
        end
    end
        
end

function [attributeindex,cond] = findbestsplitattribute(data,colsleft,classindex,method)
%if the attribute splited is not numeric, then it must be nominal.(ordinal
%attributes has already convert to numeric before passed in)
cond_ig_mat = cell(length(colsleft),2);
for ii = [1:classindex-1,classindex+1:size(data,1)]     %exclude category column,cond_ig_mat(classindex,:)=(0,0)
    [cond_ig_mat{ii,1},cond_ig_mat{ii,2}] = findbestvalueforsplitting(data,ii,classindex,method);
end

attributeindex = max(cond_ig_mat(:,2));
cond = cond_ig_mat(attributeindex,1);

end

function [value,ig] = findbestvalueforsplitting(data,colindex,classindex,method)
%CALCULATEIMPURITY calculate the ig on specify column(rows) in data
%to find the best value for splitting with specify calculate method(entropy,gini,classification error).
%
%Variable Defination:
%   data        sample data,size 1*N cell, each row stand for one observation.
%   colindex    column index for calculate impurity.
%   classindex  specify which column in data representative Category info.
%   method      'gini','entropy','classificationerror'.
%   value       value for split
%   impurity    lowest impurity based on column(rows) and value

count = getcountincol(data(:,colindex));%count distinct value for every col , iterate every distinct value for splitting

impuritybeforesplitting = calculateimpurity(data(:,colindex),method);
if isnumeric(data{1,colindex}) == true
    value_ig_mat = cell(size(count-1,1),2);
    %sort
    [~,index] = sort(cell2mat({count(:,end)}));
    count = count(index,:);
    %calculate for every internal
    for ii = 1:size(count,1)-1
        value_ig_mat{ii,1} = mean([count(ii,1),count(ii+1,1)]);
        [datal,datar] = splitdata(data,colindex,value_ig_mat{ii,1});
        value_ig_mat{ii,2} = impuritybeforesplitting-...
            +(calculateimpurity(datal(:,colindex),method)*size(datal,1)...
                + calculateimpurity(datar(:,colindex),method)*size(datar,1))/size(data,1);
    end
    
else
    %divide
    %
    %calculate for every possible split
    value_ig_mat = cell();
    for ii = 1:floor(size(count,1)/2)
        divideset = nchoosek(count(:,1),ii);
        for jj = 1:size(divideset,1)
            value_ig_mat{end+1,1} = divideset(ii,:);
            [datal,datar] = splitdata(data,colindex,value_ig_mat{end,1});
            value_ig_mat{end,2} = impuritybeforesplitting-...
            +(calculateimpurity(datal(:,colindex),method)*size(datal,1)...
                + calculateimpurity(datar(:,colindex),method)*size(datar,1))/size(data,1);
        end
    end
end

[ig,index] = max(cell2mat(value_ig_mat(:,2)));
value = value_ig_mat(index,1);


end

function set = divide(rows)%str cell

if length(rows)<=1
    return
else
    divide
end

end

function impurity = calculateimpurity(rows,method)

switch (lower(method))
    case 'gini'%for part
        %count representative a part of vlaues in data responding to classindex
        %use count=getcountincol(data(find(data(:,colindex)<=value),classindex)) to get
        calculate=@(count)(1-sum(cell2mat(count(:,end)).^2));
    case 'entropy'
        calculate=@(count)(-sum(cell2mat(count(:,end)).*log2(cell2mat(count(:,end)))));
    case 'classificationerror'
        calculate=@(count)(1-max(cell2mat(count(:,end))));
    otherwise
        disp('method not match');
end

    count = getcountincol(rows);
    impurity = calculate(count);
    
end
