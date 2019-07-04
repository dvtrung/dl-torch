// @flow
import { LOAD_SETTINGS, SAVE_SETTINGS } from '../actions/settings';
import type { Action } from './types';

const defaultState = {
  models: {},
  machines: {}
};

export default function home(state = defaultState, action: Action) {
  switch (action.type) {
    case LOAD_SETTINGS:
      return action.settings;
    default:
      return state;
  }
}
