// @flow
import { LOAD_DIRECTORY, SELECT_MODEL } from '../actions/home';
import type { Action } from './types';
import { startOfSecond } from 'date-fns';

const defaultState = {
  dirs: {},
  currentDir: 'test'
};

export default function home(state = defaultState, action: Action) {
  switch (action.type) {
    case LOAD_DIRECTORY:
      console.log({
        ...state,
        dirs: {
          ...state.dirs,
          [action.path]: {
            name: action.name,
            models: action.models
          }
        }
      });
      return {
        ...state,
        dirs: {
          ...state.dirs,
          [action.path]: {
            name: action.name,
            models: action.models
          }
        }
      };
    case SELECT_MODEL:
      const model = action.model;
      const selectedModel = {
        ...state.dirs[model.root].models[model.modelName],
        ...model
      };
      return {
        ...state,
        dirs: {
          ...state.dirs,
          [model.root]: {
            ...state.dirs[model.root],
            models: {
              ...state.dirs[model.root].models,
              [model.modelName]: selectedModel
            }
          }
        },
        selectedModel: selectedModel,
      };
    default:
      return state;
  }
}
