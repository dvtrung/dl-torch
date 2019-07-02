// @flow
import React, { Component } from 'react';
import Home from '../components/Home';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import * as HomeActions from '../actions/home';
import * as SettingActions from '../actions/settings';

function mapStateToProps(state) {
  console.log(state)
  return {
    settings: state.settings,
    dirs: state.home.dirs,
    currentDir: state.home.currentDir,
    selectedModel: state.home.selectedModel
  };
}

function mapDispatchToProps(dispatch) {
  return bindActionCreators({
    ...HomeActions, 
    ...SettingActions
  }, dispatch)
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Home);
