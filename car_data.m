clear;
fileid = fopen('car.data');
data = textscan(fileid,'%s %s %s %s %s %s %s','Delimiter',',');
tmp = cell(length(data{1}),length(data));
for ii=1:length(data)
    if isnumeric(data{ii});
        tmp(:,ii)=num2cell(data{ii});
    else
        tmp(:,ii)=data{ii};
    end
end
data=tmp;

% Attribute Values:
% 
%    buying       v-high, high, med, low
%    maint        v-high, high, med, low
%    doors        2, 3, 4, 5-more
%    persons      2, 4, more
%    lug_boot     small, med, big
%    safety       low, med, high

%    class      N          N[%]
%    -----------------------------
%    unacc     1210     (70.023 %) 
%    acc        384     (22.222 %) 
%    good        69     ( 3.993 %) 
%    v-good      65     ( 3.762 %) 

colnames = {'buying','maint','doors','persons','lug_boot','safety','class'};
classindex = length(colnames);
method = 'gini';
values{1} = {'vhigh';'high';'med';'low'};
values{2} = {'vhigh';'high';'med';'low'};
values{3} = {'2';'3';'4';'5more'};
values{4} = {'2';'4';'more'};
values{5} = {'small';'med';'big'};
values{6} = {'low';'med';'high'};
%values{7} = {'unacc';'acc';'good','vgood'};
for ii =1:length(values);
    for jj = 1:length(values{ii})
        mask = strcmp(data(:,ii),values{ii}{jj});
        data(mask,ii) = {jj};
    end
end

tree = decisiontree(data,colnames,classindex,method);
    
