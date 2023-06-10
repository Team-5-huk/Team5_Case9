import React, { useEffect, useState } from 'react';
import styles from './Account.module.scss';
import linksStore from '../../../store/linksStore';
import { apiGetUser } from '../../../api/api';
import store from '../../../store/store';
import CustomButton from '../../../components/CustomButton/CustomButton';

const Account = () => {
	const { linkGetUser } = linksStore;
	const { setAuthUser } = store;

	const [user, setUser] = useState({});

	useEffect(() => {
		getInfoUser();
	}, []);

	const getInfoUser = () => {
		apiGetUser(linkGetUser).then(({ data, error }) => {
			setUser(data);
			console.log(error);
		});
	};

	const signOut = () => {
		localStorage.removeItem('token');
		localStorage.removeItem('role');
		setAuthUser('logout');
	};

	return (
		<div>
			<div>
				{user.first_name} {user.last_name}
				{user.email}
				<CustomButton name={'Выйти'} width={'85px'} handleClick={signOut} />
			</div>
		</div>
	);
};

export default Account;
