// @flow
import {
  LOAD_DIRECTORY,
  SELECT_MODEL,
  SSH_CONNECT,
  SYNC_TRAINING_STATS_END,
  SYNC_TRAINING_STATS_START
} from '../actions/home';
import type { Action } from './types';
import {LOAD_SETTINGS} from "../actions/settings";

const defaultState = {
  dirs: {},
  machines: {},
  models: {},
  currentDir: 'test',
  selectedModelKey: null,
  selectedMachineKey: null
};

const extendModel = (state, model, value) => ({
  ...state,
  models: {
    ...state.models,
    [model.key]: {
      ...state.models[model.key],
      ...value
    }},
});

const extendMachine = (state, key, value) => ({
  ...state,
  machines: {
    ...state.machines,
    [key]: {
      ...state.machines[key],
      ...value
    }
  }
});

export default function home(state = defaultState, action: Action) {
  switch (action.type) {
    case LOAD_SETTINGS:
      const machines = Object.assign(action.settings.machines);
      Object.keys(machines).forEach((key) => machines[key].key = key);
      const models = Object.assign(action.settings.models);
      Object.keys(models).forEach((key) => models[key] = {
        ...models[key],
        key,
        stats: {
          isLoading: false
        }
      });
      return {
        ...state,
        machines,
        models,
      };
    case LOAD_DIRECTORY:
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
      return {
        ...state,
        selectedModelKey: action.key,
        selectedMachine: null
      };
    case SSH_CONNECT:
      const { machine } = action;
      return extendMachine(state, machine.key, {
        sshClient: action.client
      });
    case SYNC_TRAINING_STATS_START:
      return extendModel(state, action.model, {
        stats: { isLoading: true },
      });
    case SYNC_TRAINING_STATS_END:
      const { data } = action;
      const metrics = Object.keys(data.result)
      return extendModel(state, action.model, {
        stats: {
          isLoading: false,
          epoch: data.epoch,
          totalEpoch: 100,
          metrics: metrics,
          bestResult: metrics.map(metric => data.best_result[metric].result[metric]),
          bestResultEpoch: metrics.map(metric => data.best_result[metric].epoch),
          isLoadingStats: false
        }
      });
    default:
      return state;
  }
}
