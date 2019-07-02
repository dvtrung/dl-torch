// @flow
import { combineReducers } from 'redux';
import { connectRouter } from 'connected-react-router';
import counter from './counter';
import home from './home';
import settings from './settings'

export default function createRootReducer(history: History) {
  return combineReducers({
    router: connectRouter(history),
    counter, home, settings
  });
}
