// @flow
import {
  LOAD_DIRECTORY, SELECT_MACHINE,
  SELECT_MODEL,
  SSH_CONNECT,
  SYNC_EPOCH_STATS_END,
  SYNC_EPOCH_STATS_START,
  SYNC_EPOCH_STEP_STATS_CONNECTED,
  SYNC_EPOCH_STEP_STATS_DISCONNECTED,
  SYNC_EPOCH_STEP_STATS_START,
  SYNC_EPOCH_STEP_STATS_UPDATE
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
        },
        stepStats: {
          losses: [],
          epochs: []
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
        selectedMachineKey: null
      };
    case SELECT_MACHINE:
      return {
        ...state,
        selectedModelKey: null,
        selectedMachineKey: action.key
      };
    case SSH_CONNECT:
      const { machine } = action;
      return extendMachine(state, machine.key, {
        sshClient: action.client
      });
    case SYNC_EPOCH_STATS_START:
      return extendModel(state, action.model, {
        stats: {
          ...state.models[action.model.key].stats,
          isLoading: true
        },
      });
    case SYNC_EPOCH_STATS_END:
      const { data } = action;
      if (data.length === 0) {
        return extendModel(state, action.model, {
          stats: {
            isLoading: false
          }
        });
      } else {
        const metrics = Object.keys(data[0].result);
        return extendModel(state, action.model, {
          stats: {
            isLoading: false,
            epoch: data[data.length - 1].epoch,
            totalEpoch: 100,
            metrics: metrics,
            bestResult: metrics.map(metric => data[data.length - 1].best_result[metric].result[metric]),
            bestResultEpoch: metrics.map(metric => data[data.length - 1].best_result[metric].epoch),
            epochs: data.map(res => res.epoch),
            results: metrics.map(metric => data.map(res => res.result[metric])),
            isLoadingStats: false,
            lastUpdated: new Date()
          }
        });
      }
    case SYNC_EPOCH_STEP_STATS_START:
      return extendModel(state, action.model, {
          stepStats: {
            isConnecting: true,
            isConnected: false,
            losses: [],
            epochs: []
          }
        });
    case SYNC_EPOCH_STEP_STATS_CONNECTED:
      return extendModel(state, action.model, {
          stepStats: {
            isConnecting: false,
            isConnected: true,
            client: action.client,
            losses: [],
            epochs: []
          }
        });
    case SYNC_EPOCH_STEP_STATS_DISCONNECTED:
      return extendModel(state, action.model, {
        stepStats: {
          ...state.models[action.model.key].stepStats,
          isConnecting: false,
          isConnected: false,
          client: null
        }
      })
    case SYNC_EPOCH_STEP_STATS_UPDATE:
      if (action.data) {
        return extendModel(state, action.model, {
          stepStats: {
            ...state.models[action.model.key].stepStats,
            epoch: action.data.epoch,
            loss: action.data.loss,
            overallLoss: action.data.overall_loss,
            epochs: [...state.models[action.model.key].stepStats.epochs, action.data.epoch],
            losses: [...state.models[action.model.key].stepStats.losses, action.data.loss]
          }
        });
      }
    default:
      return state;
  }
}
