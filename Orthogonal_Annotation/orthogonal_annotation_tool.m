function varargout = orthogonal_annotation_tool(varargin)
% ORTHOGONAL_ANNOTATION_TOOL MATLAB code for orthogonal_annotation_tool.fig
%      ORTHOGONAL_ANNOTATION_TOOL, by itself, creates a new ORTHOGONAL_ANNOTATION_TOOL or raises the existing
%      singleton*.
%
%      H = ORTHOGONAL_ANNOTATION_TOOL returns the handle to a new ORTHOGONAL_ANNOTATION_TOOL or the handle to
%      the existing singleton*.
%
%      ORTHOGONAL_ANNOTATION_TOOL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ORTHOGONAL_ANNOTATION_TOOL.M with the given input arguments.
%
%      ORTHOGONAL_ANNOTATION_TOOL('Property','Value',...) creates a new ORTHOGONAL_ANNOTATION_TOOL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before orthogonal_annotation_tool_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to orthogonal_annotation_tool_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help orthogonal_annotation_tool

% Last Modified by GUIDE v2.5 08-Jun-2023 17:41:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @orthogonal_annotation_tool_OpeningFcn, ...
                   'gui_OutputFcn',  @orthogonal_annotation_tool_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before orthogonal_annotation_tool is made visible.
function orthogonal_annotation_tool_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to orthogonal_annotation_tool (see VARARGIN)

% Choose default command line output for orthogonal_annotation_tool
handles.output = hObject;

% clc;

set(handles.axes1,'xTick',[]);
set(handles.axes1,'ytick',[]);
set(handles.axes1,'box','on');
set(handles.axes2,'xTick',[]);
set(handles.axes2,'ytick',[]);
set(handles.axes2,'box','on');
set(handles.axes3,'xTick',[]);
set(handles.axes3,'ytick',[]);
set(handles.axes3,'box','on');
set(handles.axes4,'xTick',[]);
set(handles.axes4,'ytick',[]);
set(handles.axes4,'box','on');


labels_1 = zeros(128, 128);
labels_2 = zeros(128, 128);
threshold = 1;
file_path = [];
filename_in = [];
input_data = [];

handles.labels_1 = labels_1;
handles.labels_2 = labels_2;
handles.threshold = threshold;
handles.input_data = input_data;
handles.file_path = file_path;
handles.filename_in = filename_in;


% Update handles structure
guidata(hObject, handles);

% UIWAIT makes orthogonal_annotation_tool wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = orthogonal_annotation_tool_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Load_Data.
function Load_Data_Callback(hObject, eventdata, handles)
% hObject    handle to Load_Data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

threshold = handles.threshold;
input_data = handles.input_data;
labels_1 = handles.labels_1;
labels_2 = handles.labels_2;

% reset labels
labels_1 = zeros(128, 128);
labels_2 = zeros(128, 128);

% load data
[filename_in, file_path] = uigetfile('.bin', 'Choose File');

if ischar(filename_in)
    fid = fopen([file_path, filename_in], 'r');
    input_data = fread(fid, 'float');
    fclose(fid);
    input_data = reshape(input_data, [128, 128, 128]);
    % ****************************************************************** %
    % ******************* check the permute order ********************** %
%     input_data = permute(input_data, [3, 1, 2]);
    % ******************* check the permute order ********************** %
    % ****************************************************************** %
    
    % show in windows
    axes(handles.axes1);
    imagesc(input_data(:,:,64), [min(input_data(:))/threshold max(input_data(:))/threshold]);
    colormap seismic; axis image; axis off;
    axes(handles.axes2);
    imagesc(rot90(squeeze(input_data(64,:,:)),1), [min(input_data(:))/threshold max(input_data(:))/threshold]);
    colormap seismic; axis image; axis off;
    
    % reset previous label windows
    axes(handles.axes3);
    imagesc(labels_1); colormap seismic;
    axes(handles.axes4);
    imagesc(labels_2); colormap seismic;
    set(handles.axes3,'xTick',[]);
    set(handles.axes3,'ytick',[]);
    set(handles.axes3,'box','on');
    set(handles.axes4,'xTick',[]);
    set(handles.axes4,'ytick',[]);
    set(handles.axes4,'box','on'); 
end

handles.file_path = file_path;
handles.filename_in = filename_in;
handles.labels_1 = labels_1;
handles.labels_2 = labels_2;
handles.input_data = input_data;
guidata(hObject, handles);


% --- Executes on button press in Label_1.
function Label_1_Callback(hObject, eventdata, handles)
% hObject    handle to Label_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labels_1 = handles.labels_1;

% get discrete points from windows
axes(handles.axes1);

[x, y] = ginput;
x = ceil(x);
y = ceil(y);

% ind1 = find(x < 1);
% ind2 = find(y < 1);
% ind3 = find(x > 128);
% ind4 = find(y > 128);
% ind_del1 = union(ind1, ind3);
% ind_del2 = union(ind2, ind4);
% ind_del = union(ind_del1, ind_del2);
% 
% x(ind_del) = [];
% y(ind_del) = [];

x(x<1) = 1;
x(x>128) = 128;
y(y<1) = 1;
y(y>128) = 128;

% fit points to lines
[p, ~] = polyfit(y, x, 5);
label = zeros(128, 128);
yy = min(y(:)): max(y(:));
xx = polyval(p, yy);
xx = ceil(xx);
for i = 1:length(xx)
    label(yy(i), xx(i)) = 1;
end
label = label(1:128, 1:128);

% imdilate
se = strel('rectangle',[2, 2]);
label = imdilate(label,se);

% binarize
label = (label - min(label(:)))*1 / (max(label(:)) - min(label(:)));
label = imbinarize(label);

labels_1 = labels_1 + label;
labels_1 = imbinarize(labels_1);

axes(handles.axes3)
imagesc(labels_1);
axis image; axis off; colormap seismic;

handles.labels_1 = labels_1;
guidata(hObject, handles);



% --- Executes on button press in Label_2.
function Label_2_Callback(hObject, eventdata, handles)
% hObject    handle to Label_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labels_2 = handles.labels_2;

% get discrete points from windows

axes(handles.axes2);

[x, y] = ginput;
x = ceil(x);
y = ceil(y);

% ind1 = find(x < 1);
% ind2 = find(y < 1);
% ind3 = find(x > 128);
% ind4 = find(y > 128);
% ind_del1 = union(ind1, ind3);
% ind_del2 = union(ind2, ind4);
% ind_del = union(ind_del1, ind_del2);
% 
% x(ind_del) = [];
% y(ind_del) = [];

x(x<1) = 1;
x(x>128) = 128;
y(y<1) = 1;
y(y>128) = 128;

% fit points to lines

k = (max(y(:))-min(y(:)))/(max(x(:))-min(x(:)));

if k >= 1
    [p, ~] = polyfit(y, x, 7);
    label = zeros(128, 128);
    yy = min(y(:)): max(y(:));
    xx = polyval(p, yy);
    xx = ceil(xx);
    for i = 1:length(xx)
        label(yy(i), xx(i)) = 1;
    end
else
    [p, ~] = polyfit(x, y, 7);
    label = zeros(128, 128);
    xx = min(x(:)): max(x(:));
    yy = polyval(p, xx);
    yy = ceil(yy);
    for i = 1:length(yy)
        label(yy(i), xx(i)) = 1;
    end
end
label = label(1:128, 1:128);

% imdilate
se = strel('rectangle',[2, 2]);
label = imdilate(label,se);

% binarize
label = (label - min(label(:)))*1 / (max(label(:)) - min(label(:)));
label = imbinarize(label);

labels_2 = labels_2 + label;
labels_2 = imbinarize(labels_2);

axes(handles.axes4)
imagesc(labels_2);
axis image; axis off; colormap seismic;

handles.labels_2 = labels_2;
guidata(hObject, handles);



% --- Executes on button press in Save.
function Save_Callback(hObject, eventdata, handles)
% hObject    handle to Save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

labels_1 = handles.labels_1;
labels_2 = handles.labels_2;
file_path = handles.file_path;
filename_in = handles.filename_in;

if max(labels_1(:)) ~= 0 && max(labels_2(:)) ~= 0
    
    ind1 = strfind(filename_in, '_');
    ind2 = strfind(filename_in, '.');
    
    filename_out1 = [file_path, 'Label_', 'Inline64', filename_in(ind1(end): ind2(end)), 'png'];
    filename_out2 = [file_path, 'Label_', 'TimeSlice64', filename_in(ind1(end): ind2(end)), 'png'];
    imwrite(labels_1, filename_out1);
    imwrite(labels_2, filename_out2);
    msgbox('OK!');
else
    msgbox('Finish the annotation!');
end



% --- Executes when selected object is changed in uibuttongroup1.
function uibuttongroup1_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup1 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

threshold = handles.threshold;
input_data = handles.input_data;

if get(handles.threshold_norm,'value')
   threshold = 1;
elseif get(handles.threshold_mid,'value')
   threshold = 3;
elseif get(handles.threshold_high,'value')
   threshold = 5;
end

if isempty(input_data)
    msgbox('Load Data!');
else
    % refresh windows
    axes(handles.axes1);
    imagesc(input_data(:,:,64), [min(input_data(:))/threshold max(input_data(:))/threshold]);
    colormap seismic; axis image; axis off;
    axes(handles.axes2);
    imagesc(rot90(squeeze(input_data(64,:,:)),1), [min(input_data(:))/threshold max(input_data(:))/threshold]);
    colormap seismic; axis image; axis off;
end

handles.threshold = threshold;
guidata(hObject, handles);



function uibuttongroup1_CreateFcn(hObject, eventdata, handles)

function uibuttongroup1_ButtonDownFcn(hObject, eventdata, handles)

function threshold_high_Callback(hObject, eventdata, handles)

function threshold_mid_Callback(hObject, eventdata, handles)

function threshold_norm_Callback(hObject, eventdata, handles)
