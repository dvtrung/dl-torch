import type { Dispatch as ReduxDispatch, Store as ReduxStore } from 'redux';

export type counterStateType = {
  +counter: number
};

export type Machine = {
  title: string,
  host: string,
  username: string,
  privateKey: string,
  root: string,
  tmpPath: string,
};

export type Action = {
  +type: string
};

export type Model = {
  path: string,
  name: string
};

export type GetState = () => counterStateType;

export type Dispatch = ReduxDispatch<Action>;

export type Store = ReduxStore<GetState, Action>;
