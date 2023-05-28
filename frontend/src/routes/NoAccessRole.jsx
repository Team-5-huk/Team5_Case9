import React from 'react';
import {Routes, Route} from 'react-router-dom';
import Login from '../pages/Login/Login';
import Redirect from '../helpers/Redirect';

const NoAccessRole = () => {
  return (
    <Routes>
        <Route path='/sigin' element={<Login />}/>
        <Route path='*' element={<Redirect to='/sigin' />} />
    </Routes>
  )
}

export default NoAccessRole;