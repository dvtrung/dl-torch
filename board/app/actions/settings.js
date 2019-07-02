// @flow
import type { GetState, Dispatch } from '../reducers/types';
const path = require('path');
const fs = require('fs');

export const LOAD_SETTINGS = 'LOAD_SETTINGS';
export const SAVE_SETTINGS = 'SAVE_SETTINGS';

export function loadSettings() {
  return (dispatch: Dispatch) => {
    const content = fs.readFileSync(path.join(__dirname, "settings.json"));
    dispatch({
      type: LOAD_SETTINGS,
      settings: JSON.parse(content)
    });
  }
}
