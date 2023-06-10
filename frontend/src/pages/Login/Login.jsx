import React from 'react';
import styles from './Login.module.scss';
import store from '../../store/store';
import { Box } from '@mui/material';
import linksStore from '../../store/linksStore';
import CustomButton from '../../components/CustomButton/CustomButton';
import { apiPostAuthorize } from '../../api/api';
import { observer } from 'mobx-react-lite';

const Login = observer(() => {
	const { setAuthUser } = store;
	const { linkLogin } = linksStore;

	const loginApi = (email, password) => {
		apiPostAuthorize(linkLogin, { email, password }).then(({ data, error }) => {
			console.log(error);

			setAuthUser(data.role);
			localStorage.setItem('role', data.role);
			localStorage.setItem('token', data.token);
		});
	};

	return (
		<Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
			<div className={styles.title}>Войти в аккаунт</div>

			<Box>
				{/* <div style={{ margin: '20px' }}>
					<CustomButton name="Пользователь" handleClick={() => loginApi('user@user.ru', 'user')} />
				</div> */}

				<div style={{ margin: '20px' }}>
					<CustomButton name="Обходчик" width={'140px'} handleClick={() => loginApi('employee@employee.ru', 'employee')} />
				</div>

				<div style={{ margin: '20px' }}>
					<CustomButton name="Администратор" width={'140px'} handleClick={() => loginApi('admin@admin.ru', 'admin')} />
				</div>
			</Box>
		</Box>
	);
});

export default Login;
