function mdlInitializeSizes(s)
ssSetNumInputPorts(s, 3);  % 입력 포트 수를 3개로 설정
ssSetInputPortWidth(s, 0, 1); % 첫 번째 입력 포트의 폭 설정
ssSetInputPortWidth(s, 1, 1); % 두 번째 입력 포트의 폭 설정
ssSetInputPortWidth(s, 2, 1); % 세 번째 입력 포트의 폭 설정
end
